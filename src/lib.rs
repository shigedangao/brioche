#![recursion_limit = "256"]
use crate::network::decoder::{
    Decoder, DecoderType,
    multires_conv::{MultiResConv, MultiResDecoderConfig},
};
use crate::network::{
    Network, NetworkConfig,
    encoder::{Encoder, EncoderConfig},
    fov::{Fov, FovConfig},
};
#[cfg(feature = "burn_onnx")]
use crate::vit::{common_burn::CommonVitModel, patch_burn::PatchVitModel};
use anyhow::{Result, anyhow};
use brioche_seq::{BriocheHeadConfig, BriocheSeq};
#[cfg(feature = "f16")]
use burn::tensor::f16;
use burn::{
    Tensor,
    backend::wgpu::FloatElement,
    nn::interpolate::{Interpolate2dConfig, InterpolateMode},
    prelude::{Backend, Module},
};
use std::f32::consts::PI;
#[cfg(feature = "ort_onnx")]
use {
    crate::vit::{common::CommonVitModel, patch::PatchVitModel},
    ort::value::PrimitiveTensorElementType,
};

mod brioche_seq;
pub mod four;
mod network;
mod utils;
mod vit;

#[cfg(feature = "burn_onnx")]
mod model;

/// `MixedFloats` is a trait that defines a type that can be used as a placeholder to support F32 & F16 float types.
#[cfg(feature = "ort_onnx")]
pub trait MixedFloats: FloatElement + PrimitiveTensorElementType {}
#[cfg(feature = "burn_onnx")]
pub trait MixedFloats: FloatElement {}

// Blanket implementation
impl MixedFloats for f32 {}
impl MixedFloats for f64 {}
#[cfg(feature = "f16")]
impl MixedFloats for f16 {}

// Constants
const CLAMP_MIN: f32 = 1e-4;
const CLAMP_MAX: f32 = 1e4;

/// Brioche is a struct which implements the Depth-pro main class. The implementation refer to the one below
///
/// @link <https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/depth_pro.py#L157>
#[derive(Debug, Module)]
pub struct Brioche<B: Backend> {
    head: BriocheSeq<B>,
    encoder: Encoder<B>,
    decoder: MultiResConv<B>,
    fov: Fov<B>,
}

impl<B: Backend> Brioche<B> {
    /// Creates a new `Brioche` instance with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `encoder_config` - Configuration for the encoder.
    /// * `decoder_config` - Configuration for the decoder.
    /// * `fov_config` - Configuration for the fov.
    /// * `head_config` - Configuration for the head.
    /// * `device` - The device to use for the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be created.
    ///
    /// # Returns
    ///
    /// A new `Brioche` instance.
    pub fn new(
        encoder_config: EncoderConfig,
        decoder_config: MultiResDecoderConfig,
        fov_config: FovConfig,
        head_config: BriocheHeadConfig,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            head: BriocheSeq::<B>::new(NetworkConfig::Head(head_config), device)?,
            encoder: Encoder::<B>::new(NetworkConfig::Encoder(encoder_config), device)?,
            decoder: MultiResConv::<B>::new(NetworkConfig::Decoder(decoder_config), device)?,
            fov: Fov::<B>::new(NetworkConfig::Fov(fov_config), device)?,
        })
    }

    /// With record load weights from the given paths into the model.
    ///
    /// # Arguments
    ///
    /// * `decoder_weight_path` - Path to the decoder weights.
    /// * `encoder_weight_path` - Path to the encoder weights.
    /// * `fov_weight_path` - Path to the fov weights.
    /// * `head_weight_path` - Path to the head weights.
    /// * `device` - Device to load the weights onto.
    ///
    /// # Error
    ///
    /// Returns an error if the weights cannot be loaded from the given paths.
    ///
    /// # Result
    ///
    /// Returns the updated model with the weights loaded from the given paths.
    pub fn with_record<S: AsRef<str>>(
        mut self,
        decoder_weight_path: S,
        encoder_weight_path: S,
        fov_weight_path: S,
        head_weight_path: S,
    ) -> Result<Self> {
        // Load weights from the decoder path into the model.
        self.decoder = self.decoder.with_record(decoder_weight_path.as_ref())?;

        // Load weights from the encoder path into the model.
        self.encoder = self.encoder.with_record(encoder_weight_path.as_ref())?;

        // Load weights from the fov path into the model.
        self.fov = self.fov.with_record(fov_weight_path.as_ref())?;

        // Load weights from the head path into the model.
        self.head = self.head.with_record(head_weight_path.as_ref())?;

        Ok(self)
    }

    /// Infer the model for the given input tensor.
    /// /!\ For a trial implementation the "`f_px`" parameter is not taken into account.
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
        inputs: (Tensor<B, 3>, Tensor<B, 3>),
        patch_encoder: PatchVitModel,
        image_encoder: CommonVitModel,
        img_size: usize,
        device: &B::Device,
    ) -> Result<(Tensor<B, 2>, Option<Tensor<B, 4>>)> {
        let (input, fov_x) = inputs;

        // Squeeze the tensor on the 0 dimension
        let x: Tensor<B, 4> = input.unsqueeze_dim(0);
        // Perform the same squeeze on the fov input tensor
        let [_, _, h, w] = x.shape().dims();

        // If the image size is different then we need to resize the input tensor
        let (interpolated_tensor, resize) = match h != img_size || w != img_size {
            true => {
                let interpolation = Interpolate2dConfig::new()
                    .with_mode(InterpolateMode::Linear)
                    .with_output_size(Some([img_size, img_size]))
                    .init();

                (interpolation.forward(x), true)
            }
            false => (x, false),
        };

        let (canonical_inverse_depth, fov_deg) = self
            .forward::<F>(
                (interpolated_tensor, fov_x),
                patch_encoder,
                image_encoder,
                img_size,
                device,
            )
            .map_err(|err| anyhow!("Unable to perform the forward of the model due to {err}"))?;

        let fov_deg_to_rad = fov_deg * PI / 180.;
        let f_px = 0.5 * w as f32 / (fov_deg_to_rad * 0.5).tan();
        let mut inverse_depth = canonical_inverse_depth * (w as f32 / f_px.clone());

        let f_px_squeeze = match f_px.shape().dims() != [1, 1, 1, 1] {
            true => Some(f_px.squeeze()),
            false => None,
        };

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
    /// * `input` - Input tensor of shape [`batch_size`, `channels`, `height`, `width`].
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
        inputs: (Tensor<B, 4>, Tensor<B, 3>),
        patch_encoder: PatchVitModel,
        image_encoder: CommonVitModel,
        img_size: usize,
        device: &B::Device,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>)> {
        let (input, fov_input) = inputs;

        let [_, _, h, w] = input.shape().dims();
        if h != img_size || w != img_size {
            return Err(anyhow!("input image size does not match the expected size"));
        }

        let encodings = self
            .encoder
            .forward::<F>(input, patch_encoder, image_encoder, device)?;

        let (features, features_0) = self.decoder.forward(DecoderType::MultiResConv(vec![
            encodings.x_latent0,
            encodings.x_latent1,
            encodings.x0,
            encodings.x1,
            encodings.x,
        ]))?;

        let canonical_inverse_depth = self.head.forward(features);
        if features_0.is_none() {
            return Err(anyhow!("features_0 is None"));
        }

        let fov_deg = self.fov.forward(fov_input, features_0.unwrap());

        Ok((canonical_inverse_depth, fov_deg))
    }
}
