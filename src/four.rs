use crate::MixedFloats;
use crate::brioche_seq::BriocheHeadConfig;
use crate::network::decoder::multires_conv::MultiResDecoderConfig;
use crate::network::encoder::EncoderConfig;
use crate::network::fov::FovConfig;
use crate::utils;
use crate::vit::{common::CommonVitModel, patch::PatchVitModel};
use crate::{Brioche, network::Network};
use anyhow::{Result, anyhow};
use burn::{
    prelude::Backend,
    tensor::{Tensor, Transaction},
};
use image::{ImageBuffer, Rgb};
use ndarray::{Array, Array2};
use ndarray_stats::QuantileExt;
use std::path::PathBuf;

// Constants
const LAST_DIMS: (usize, usize) = (31, 1);
const DIM_DECODER: usize = 256;
const DIM_ENCODER: [usize; 4] = [256, 512, 1024, 1024];
const EMBED_DIM: usize = 1024;
const ENCODER_IMG_SIZE: usize = 384 * 4;

/// Runner is a struct which helps to run the depth-pro model
pub struct Four<B: Backend> {
    model: Brioche<B>,
    image_model: CommonVitModel,
    fov_model: CommonVitModel,
    patch_model: PatchVitModel,
    device: B::Device,
}

impl<B: Backend> Four<B> {
    /// Create a new four.
    ///
    /// # Arguments
    /// * `fov_encoder_path` - Path to the fov encoder model
    /// * `patch_vit_path` - Path to the patch vit model
    /// * `image_vit_path` - Path to the image vit model
    /// * `vit_thread_nb` - Number of threads to use for vit models
    /// * `device` - Device to use for the model
    pub fn new<S: AsRef<str>>(
        fov_encoder_path: S,
        patch_vit_path: S,
        image_vit_path: S,
        vit_thread_nb: usize,
        fov_weight_path: S,
        encoder_weight_path: S,
        decoder_weight_path: S,
        head_weight_path: S,
    ) -> Result<Self> {
        let fov_model =
            CommonVitModel::new(PathBuf::from(fov_encoder_path.as_ref()), vit_thread_nb)?;

        let patch_model =
            PatchVitModel::new(PathBuf::from(patch_vit_path.as_ref()), vit_thread_nb)?;

        let image_model =
            CommonVitModel::new(PathBuf::from(image_vit_path.as_ref()), vit_thread_nb)?;

        // @TODO make it as a parameters if the model work...
        let encoder_config = EncoderConfig {
            dims_encoder: vec![256, 512, 1024, 1024],
            patch_encoder_embed_dim: EMBED_DIM,
            image_encoder_embed_dim: EMBED_DIM,
            decoder_features: DIM_DECODER,
            out_size: 384 / 16,
        };

        let decoder_config = MultiResDecoderConfig {
            dims_encoder: vec![vec![DIM_DECODER], DIM_ENCODER.to_vec()].concat(),
            dim_decoder: DIM_DECODER,
        };

        let fov_config = FovConfig {
            num_features: DIM_DECODER,
            with_fov_encoder: true,
            embed_dim: EMBED_DIM,
        };

        let brioche_head_config = BriocheHeadConfig {
            last_dims: LAST_DIMS,
            dim_decoder: DIM_DECODER,
        };

        let device = Default::default();

        // Create the brioche (depth-pro)model
        let mut bm = Brioche::<B>::new(
            encoder_config,
            decoder_config,
            fov_config,
            brioche_head_config,
            &device,
        )?;

        // Set the weights on the property of the model.
        bm.decoder = bm.decoder.with_record(decoder_weight_path, &device);
        bm.encoder = bm.encoder.with_record(encoder_weight_path, &device);
        bm.fov = bm.fov.with_record(fov_weight_path, &device);
        bm.head = bm.head.with_record(head_weight_path, &device);

        Ok(Self {
            model: bm,
            fov_model,
            image_model,
            patch_model,
            device,
        })
    }

    /// Four run the brioche (depth-pro) model and export the output image into a buffer
    pub fn run<S: AsRef<str>, F: MixedFloats>(
        mut self,
        image_path: S,
        is_half_precision: bool,
    ) -> Result<(ImageBuffer<Rgb<u8>, Vec<u8>>, Option<Tensor<B, 4>>)> {
        let img = image::open(PathBuf::from(image_path.as_ref()))
            .map_err(|err| anyhow!("Unable to load the image due to {err}"))?;
        let input = utils::preprocess_image::<B>(&img, &self.device, is_half_precision)
            .map_err(|err| anyhow!("Unable to preprocess the image due to {err}"))?;

        dbg!("input tensor generation done");

        let (depth, focallength_px) = self.model.infer::<F>(
            input,
            self.patch_model,
            self.image_model,
            self.fov_model,
            ENCODER_IMG_SIZE,
            &self.device,
        )?;

        let [h, w]: [usize; 2] = depth.shape().dims();
        let tensor_data = Transaction::default().register(depth).execute();

        let depth_tensor_data: Vec<f32> = match tensor_data.first() {
            Some(d) => d.to_vec().map_err(|err| {
                anyhow!("Unable to convert the tensor to a vector due to {:?}", err)
            })?,
            None => {
                return Err(anyhow!(
                    "Unable to convert the tensor to a vector due to {:?}",
                    tensor_data
                ));
            }
        };

        // Use ndarray in order to perform the operation
        let squeezed_depth = Array::from_shape_vec((h, w), depth_tensor_data)?
            .into_dyn()
            .squeeze();

        let inverse_depth = 1. / squeezed_depth;
        let inverse_depth_max = inverse_depth
            .max()
            .map_err(|err| anyhow!("Expect to get the max value from the tensor: {err}"))?;
        let inverse_depth_min = inverse_depth
            .min()
            .map_err(|err| anyhow!("Expect to get the min value from the tensor: {err}"))?;

        // Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        let max_invdepth_vizu = inverse_depth_max.min(1. / 0.1);
        let min_invdepth_vizu = inverse_depth_min.max(1. / 250.);

        let inverse_depth_normalized =
            (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu);

        // Get the shape of the inverse_depth
        let idn_shape = inverse_depth_normalized.shape();
        if idn_shape.len() < 2 {
            return Err(anyhow!("Expect final shape to be superior to 2"));
        }
        let (height, width) = (
            idn_shape.first().copied().unwrap_or_default(),
            idn_shape.get(1).copied().unwrap_or_default(),
        );

        // Normalized the matrix to a defined shape of two (heigh, width).
        let inverse_depth_normalized_normalized: Array2<f32> = inverse_depth_normalized
            .map(|v| v.clamp(0., 1.0))
            .into_shape_with_order((height, width))?;

        let cmap_matrix = utils::cmap(&inverse_depth_normalized_normalized);
        let cmap_matrix = utils::drop_alpha(cmap_matrix);

        let (raw_vec, _) = cmap_matrix.into_raw_vec_and_offset();
        let img_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width as u32, height as u32, raw_vec).unwrap();

        Ok((img_buffer, focallength_px))
    }
}
