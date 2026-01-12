#![recursion_limit = "256"]
use crate::network::decoder::multires_conv::{MultiResConv, MultiResDecoderConfig};
use crate::network::decoder::{Decoder, DecoderType};
use crate::network::encoder::{Encoder, EncoderConfig};
use crate::network::fov::{Fov, FovConfig};
use crate::vit::common::CommonVitModel;
use crate::vit::patch::PatchVitModel;
use anyhow::{Result, anyhow};
use brioche_seq::BriocheSeq;
use burn::Tensor;
use burn::prelude::{Backend, Module};

mod brioche_seq;
mod network;
mod vit;

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
            encoder: Encoder::<B>::new(encoder_config, &device),
            decoder: MultiResConv::<B>::new(decoder_config, &device),
            fov: Fov::<B>::new(fov_config, &device),
        })
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
        device: &B::Device,
        patch_encoder: PatchVitModel,
        image_encoder: CommonVitModel,
        fov_image_encoder: CommonVitModel,
        img_size: usize,
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
