use crate::MixedFloats;
use crate::brioche_seq::BriocheHeadConfig;
use crate::network::decoder::multires_conv::MultiResDecoderConfig;
use crate::network::encoder::EncoderConfig;
use crate::network::fov::FovConfig;
use crate::utils;
use crate::vit::patch::PatchVitModel;
use crate::{Brioche, network::Network};
use anyhow::{Result, anyhow};
use burn::tensor::Transaction;
use burn::{prelude::Backend, tensor::Tensor};
use image::{ImageBuffer, Rgb};
use ndarray::{Array, Array2};
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
        patch_vit_path: S,
        vit_thread_nb: usize,
        fov_weight_path: S,
        encoder_weight_path: S,
        decoder_weight_path: S,
        head_weight_path: S,
    ) -> Result<Self> {
        let patch_model =
            PatchVitModel::new(PathBuf::from(patch_vit_path.as_ref()), vit_thread_nb)?;

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

        let (depth, focallength_px) =
            self.model
                .infer::<F>(input, self.patch_model, ENCODER_IMG_SIZE, &self.device)?;

        let [h, w]: [usize; 2] = depth.shape().dims();
        let squeezed_depth: Tensor<B, 2> = depth.detach().squeeze();

        dbg!("finish extract tensor");

        let inverse_depth: Tensor<B, 2> = 1. / squeezed_depth;
        let inverse_depth_min_tensor = inverse_depth.clone().min();
        let inverse_depth_max_tensor = inverse_depth.clone().max();

        let extracted_tensor = Transaction::default()
            .register(inverse_depth_min_tensor)
            .register(inverse_depth_max_tensor)
            .execute()
            .to_vec();

        let Some(inverse_depth_min) = extracted_tensor
            .get(0)
            .and_then(|t| t.to_vec::<f32>().ok())
            .and_then(|v| v.first().copied())
        else {
            return Err(anyhow!("Unable to get the inverse depth min"));
        };

        let Some(inverse_depth_max) = extracted_tensor
            .get(1)
            .and_then(|t| t.to_vec::<f32>().ok())
            .and_then(|v| v.first().copied())
        else {
            return Err(anyhow!("Unable to get the inverse depth max"));
        };

        // Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        let min_invdepth_vizu = inverse_depth_min.max(1. / 250.);
        let max_invdepth_vizu = inverse_depth_max.min(1. / 0.1);

        let inverse_depth_normalized =
            (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu);

        // Get the shape of the inverse_depth
        let [height, width] = inverse_depth_normalized.shape().dims();

        let inverse_depth_normalized_tensor = inverse_depth_normalized.clamp(0., 1.);
        let inverse_depth_normalized = inverse_depth_normalized_tensor
            .to_data()
            .into_vec::<f32>()
            .map(|v| Array::from_shape_vec((h, w), v))??;

        // Normalized the matrix to a defined shape of two (heigh, width).
        let inverse_depth_normalized: Array2<f32> =
            inverse_depth_normalized.into_shape_with_order((height, width))?;

        let cmap_matrix = utils::cmap(&inverse_depth_normalized);
        let cmap_matrix = utils::drop_alpha(cmap_matrix);

        let (raw_vec, _) = cmap_matrix.into_raw_vec_and_offset();
        let img_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width as u32, height as u32, raw_vec).unwrap();

        Ok((img_buffer, focallength_px))
    }
}
