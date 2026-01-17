#![recursion_limit = "256"]
use crate::network::decoder::multires_conv::{MultiResConv, MultiResDecoderConfig};
use crate::network::decoder::{Decoder, DecoderType};
use crate::network::encoder::{Encoder, EncoderConfig};
use crate::network::fov::{Fov, FovConfig};
use crate::network::{Network, NetworkConfig};
use crate::vit::common::CommonVitModel;
use crate::vit::patch::PatchVitModel;
use anyhow::{Result, anyhow};
use brioche_seq::BriocheSeq;
use burn::Tensor;
use burn::nn::interpolate::{Interpolate1dConfig, Interpolate2dConfig, InterpolateMode};
use burn::prelude::{Backend, Module};
use utils::DegToRad;

mod brioche_seq;
pub mod four;
mod network;
mod utils;
mod vit;

// Constants
const CLAMP_MIN: f32 = 1e-4;
const CLAMP_MAX: f32 = 1e4;

/// Brioche is a struct which implements the Depth-pro main class. The implementation refer to the one below
///
/// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/depth_pro.py#L157
#[derive(Debug, Module)]
pub struct Brioche<B: Backend> {
    head: BriocheSeq<B>,
    encoder: Encoder<B>,
    decoder: MultiResConv<B>,
    fov: Fov<B>,
}

impl<B: Backend> Brioche<B> {
    pub fn new(
        last_dims: (usize, usize),
        dim_decoder: usize,
        encoder_config: EncoderConfig,
        decoder_config: MultiResDecoderConfig,
        fov_config: FovConfig,
        device: &B::Device,
    ) -> Result<Self> {
        let head = BriocheSeq::new(dim_decoder, last_dims, device);

        Ok(Self {
            head,
            encoder: Encoder::<B>::new(NetworkConfig::Encoder(encoder_config), &device)?,
            decoder: MultiResConv::<B>::new(NetworkConfig::Decoder(decoder_config), &device)?,
            fov: Fov::<B>::new(NetworkConfig::Fov(fov_config), &device)?,
        })
    }

    /// Infer the model for the given input tensor.
    /// /!\ For a trial implementation the "f_px" parameter is not taken into account.
    ///
    /// # Arguments
    /// * `input` - The input tensor.
    /// * `patch_encoder` - The patch encoder model.
    /// * `image_encoder` - The image encoder model.
    /// * `fov_image_encoder` - The field of view image encoder model.
    /// * `img_size` - The image size.
    /// * `device` - The device.
    ///
    /// # Returns
    /// * `Tensor<B, 4>` - The depth tensor.
    /// * `Tensor<B, 4>` - The field of view tensor.
    pub fn infer(
        &mut self,
        input: Tensor<B, 3>,
        patch_encoder: PatchVitModel,
        image_encoder: CommonVitModel,
        fov_image_encoder: CommonVitModel,
        img_size: usize,
        device: &B::Device,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
        let [_, _, h, w] = input.shape().dims();
        // If the size is different then we need to resize the input tensor
        let (interpolated_tensor, resize) = match h != img_size || w != img_size {
            true => {
                let interpolatation = Interpolate1dConfig::new()
                    .with_mode(InterpolateMode::Linear)
                    .with_output_size(Some(img_size))
                    .init();

                (interpolatation.forward(input), true)
            }
            false => (input, false),
        };

        // Image tensor usually have the component [C, H, W]
        let unsqueeze_interpolate_tensor: Tensor<B, 4> = interpolated_tensor.unsqueeze();

        let (canonical_inverse_depth, fov_deg) = self.forward(
            unsqueeze_interpolate_tensor,
            patch_encoder,
            image_encoder,
            fov_image_encoder,
            img_size,
            device,
        )?;

        let mut deg_to_rad_handler = DegToRad {};
        let fov_deg_to_rad = fov_deg.map(&mut deg_to_rad_handler) * 0.5;

        let f_px = 0.5 * w as f32 / fov_deg_to_rad.tan();
        let mut inverse_depth = canonical_inverse_depth * (w as f32 / f_px.clone());
        let f_px_squeeze: Tensor<B, 4> = f_px.squeeze();

        if resize {
            let inverse_depth_interpolate_fn = Interpolate2dConfig::new()
                .with_output_size(Some([h, w]))
                .with_mode(InterpolateMode::Linear)
                .init();

            inverse_depth = inverse_depth_interpolate_fn.forward(inverse_depth);
        }

        let depth: Tensor<B, 4> = 1. / inverse_depth.clamp(CLAMP_MIN, CLAMP_MAX);

        Ok((depth.squeeze(), f_px_squeeze))
    }

    /// Forward pass of the Brioche model.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, channels, height, width].
    /// * `device` - Device to run the model on.
    /// * `patch_encoder` - Patch encoder model.
    /// * `image_encoder` - Image encoder model.
    /// * `fov_image_encoder` - Field of view image encoder model.
    /// * `img_size` - Size of the input image.
    ///
    /// # Returns
    /// * `canonical_inverse_depth` - Canonical inverse depth tensor of shape [batch_size, channels, height, width].
    /// * `fov_deg` - Field of view angle tensor of shape [batch_size, channels, height, width].
    pub fn forward(
        &mut self,
        input: Tensor<B, 4>,
        patch_encoder: PatchVitModel,
        image_encoder: CommonVitModel,
        fov_image_encoder: CommonVitModel,
        img_size: usize,
        device: &B::Device,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
        let [_, _, h, w] = input.shape().dims();
        if h != img_size * 4 || w != img_size * 4 {
            return Err(anyhow!("input image size does not match the expected size"));
        }

        let encodings =
            self.encoder
                .forward(input.clone(), patch_encoder, image_encoder, device)?;

        let (features, features_0) = self.decoder.forward(DecoderType::MultiResConv(vec![
            encodings.x_latent0_features,
            encodings.x_latent1_features,
            encodings.x0_features,
            encodings.x1_features,
            encodings.x_global_features,
        ]))?;

        let canonical_inverse_depth = self.head.forward(features);
        if features_0.is_none() {
            return Err(anyhow!("features_0 is None"));
        }

        let fov_deg = self
            .fov
            .forward(input, features_0.unwrap(), fov_image_encoder, &device)?;

        Ok((canonical_inverse_depth, fov_deg))
    }
}
