use crate::network::decoder::multires_conv::MultiResDecoderConfig;
use crate::network::encoder::EncoderConfig;
use crate::network::fov::FovConfig;
use crate::utils;
use crate::vit::common::CommonVitModel;
use crate::vit::patch::PatchVitModel;
use crate::{Brioche, network::Network};
use anyhow::{Result, anyhow};
use burn::prelude::Backend;
use colorgrad::Gradient;
use image::RgbImage;
use ndarray::Array;
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
        decoder_weight_path: S,
        encoder_weight_path: S,
        fov_weight_path: S,
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

        let device = Default::default();

        // Create the brioche (depth-pro)model
        let mut bm = Brioche::<B>::new(
            LAST_DIMS,
            DIM_DECODER,
            encoder_config,
            decoder_config,
            fov_config,
            &device,
        )?;

        // Set the weights on the property of the model.
        bm.decoder = bm.decoder.with_record(decoder_weight_path, &device);
        bm.encoder = bm.encoder.with_record(encoder_weight_path, &device);
        bm.fov = bm.fov.with_record(fov_weight_path, &device);

        Ok(Self {
            model: bm,
            fov_model,
            image_model,
            patch_model,
            device,
        })
    }

    /// Four run the brioche (depth-pro) model and export the output image into a buffer
    pub fn run<S: AsRef<str>>(mut self, image_path: S) -> Result<()> {
        let img = image::open(PathBuf::from(image_path.as_ref()))?;
        let input = utils::preprocess_image::<B>(&img, &self.device)?;

        let (depth, focallength_px) = self.model.infer(
            input,
            self.patch_model,
            self.image_model,
            self.fov_model,
            ENCODER_IMG_SIZE,
            &self.device,
        )?;
        dbg!(focallength_px);

        let [b, c, h, w]: [usize; 4] = depth.shape().dims();
        let depth_tensor_data: Vec<f64> = depth.to_data().to_vec()?;

        // Use ndarray in order to perform the operation
        let squeezed_depth = Array::from_shape_vec((b, c, h, w), depth_tensor_data)?
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

        // Create the turbo gradient
        let gradient = colorgrad::preset::turbo();

        // Create an RGB image buffer
        let mut img_buffer = RgbImage::new(width as u32, height as u32);

        // Map each normalized depth value to a color
        for y in 0..height {
            for x in 0..width {
                let value = inverse_depth_normalized[[y, x]];

                // Clamp value to [0, 1] range to be safe
                let clamped_value = value.clamp(0.0, 1.0);

                // Get the color from the gradient
                let color = gradient.at(clamped_value as f32);
                let rgba = color.to_rgba8();

                // Set the pixel (taking only RGB, ignoring alpha)
                img_buffer.put_pixel(x as u32, y as u32, image::Rgb([rgba[0], rgba[1], rgba[2]]));
            }
        }

        // Save the image
        let color_map_output_file = format!("{}.jpg", "test");
        img_buffer
            .save_with_format(&color_map_output_file, image::ImageFormat::Jpeg)
            .map_err(|err| anyhow!("Failed to save color-mapped depth: {err}"))?;

        Ok(())
    }
}
