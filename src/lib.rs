#![recursion_limit = "256"]
use crate::model::{fov_model::Model as FovModel, image_model::Model as ImageModel};
use crate::network::decoder::{
    Decoder, DecoderType,
    multires_conv::{MultiResConv, MultiResDecoderConfig},
};
use crate::network::encoder::{Encoder, EncoderConfig};
use crate::network::fov::{Fov, FovConfig};
use crate::network::{Network, NetworkConfig};
use crate::vit::patch::PatchVitModel;
use anyhow::{Result, anyhow};
use brioche_seq::{BriocheHeadConfig, BriocheSeq};
use burn::Tensor;
use burn::backend::wgpu::FloatElement;
use burn::nn::interpolate::{Interpolate2dConfig, InterpolateMode};
use burn::prelude::{Backend, Module};
use burn::tensor::{DType, f16};
use ort::tensor::PrimitiveTensorElementType;
use std::f32::consts::PI;

mod brioche_seq;
pub mod four;
mod model;
mod network;
mod utils;
mod vit;

/// MixedFloats is a trait that defines a type that can be used as a placeholder to support F32 & F16 float types.
pub trait MixedFloats: FloatElement + PrimitiveTensorElementType {}

// Blanket implementation
impl MixedFloats for f32 {}
impl MixedFloats for f64 {}
impl MixedFloats for f16 {}

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
        encoder_config: EncoderConfig,
        decoder_config: MultiResDecoderConfig,
        fov_config: FovConfig,
        head_config: BriocheHeadConfig,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            head: BriocheSeq::<B>::new(NetworkConfig::Head(head_config), &device)?,
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
    pub fn infer<F: MixedFloats>(
        &mut self,
        input: Tensor<B, 3>,
        patch_encoder: PatchVitModel,
        img_size: usize,
        device: &B::Device,
    ) -> Result<(Tensor<B, 2>, Option<Tensor<B, 4>>)> {
        // Squeeze the tensor on the 0 dimension
        let x: Tensor<B, 4> = input.unsqueeze_dim(0);
        let [_, _, h, w] = x.shape().dims();

        // If the image size is different then we need to resize the input tensor
        let (interpolated_tensor, resize) = match h != img_size || w != img_size {
            true => {
                let interpolatation = Interpolate2dConfig::new()
                    .with_mode(InterpolateMode::Linear)
                    .with_output_size(Some([img_size, img_size]))
                    .init();

                (interpolatation.forward(x), true)
            }
            false => (x, false),
        };

        let (canonical_inverse_depth, fov_deg) = self
            .forward::<F>(interpolated_tensor, patch_encoder, img_size, device)
            .map_err(|err| anyhow!("Unable to perform the forward of the model due to {err}"))?;

        let fov_deg_to_rad = fov_deg * PI / 180.;
        let f_px = 0.5 * w as f32 / (fov_deg_to_rad * 0.5).tan();
        let mut inverse_depth = canonical_inverse_depth * (w as f32 / f_px.clone());

        let mut f_px_squeeze = None;
        if f_px.shape().dims() != [1, 1, 1, 1] {
            f_px_squeeze = Some(f_px.squeeze());
        }

        dbg!("fpx squeeze");

        if resize {
            let inverse_depth_interpolate_fn = Interpolate2dConfig::new()
                .with_output_size(Some([h, w]))
                .with_mode(InterpolateMode::Linear)
                .init();

            inverse_depth = inverse_depth_interpolate_fn.forward(inverse_depth);
        }

        dbg!("resize done");

        let depth: Tensor<B, 4> = 1. / inverse_depth.clamp(CLAMP_MIN, CLAMP_MAX);

        dbg!("clamp");

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
    pub fn forward<F: MixedFloats>(
        &mut self,
        input: Tensor<B, 4>,
        patch_encoder: PatchVitModel,
        img_size: usize,
        device: &B::Device,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
        let [_, _, h, w] = input.shape().dims();
        if h != img_size || w != img_size {
            return Err(anyhow!("input image size does not match the expected size"));
        }

        dbg!("performing encoder");

        let encodings = self.encoder.forward::<F>(
            input.clone(),
            patch_encoder,
            ImageModel::new(device),
            device,
        )?;

        dbg!("encoder finish");

        let (features, features_0) = self.decoder.forward(DecoderType::MultiResConv(vec![
            encodings.x_latent0_features,
            encodings.x_latent1_features,
            encodings.x0_features,
            encodings.x1_features,
            encodings.x_global_features,
        ]))?;

        dbg!("decoder finished");

        let canonical_inverse_depth = self.head.forward(features);
        if features_0.is_none() {
            return Err(anyhow!("features_0 is None"));
        }

        dbg!("head forward done");

        let is_half = match input.dtype() {
            DType::F16 => true,
            _ => false,
        };

        let fov_deg = self
            .fov
            .forward::<F>(
                input,
                features_0.unwrap().detach(),
                is_half,
                FovModel::new(device),
            )
            .map_err(|err| anyhow!("Unable to perform forward on the fov: {err}"))?;

        dbg!("fov done");

        Ok((canonical_inverse_depth, fov_deg))
    }
}
