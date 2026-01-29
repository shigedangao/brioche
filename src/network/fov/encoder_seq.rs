use crate::MixedFloats;
use crate::vit::{VitOps, common::CommonVitModel};
use anyhow::Result;
use burn::{
    Tensor,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
};

/// Sequential Foveal Vision Transformer (FOV) network encoder.
///
/// This module implements the following fov part of depth-pro. refer to the link below
/// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L48
///
/// /!\ Note that this is tightly linked to the fov_encoder module that needs to be converted with onnx.
#[derive(Debug, Module)]
pub struct SequentialFovNetworkEncoder<B: Backend> {
    pub linear: Linear<B>,
}

impl<B: Backend> SequentialFovNetworkEncoder<B> {
    /// Create a new SequentialFovNetworkEncoder module.
    ///
    /// # Arguments
    /// * `embed_dim` - The input dimension.
    /// * `num_features` - The output dimension.
    /// * `device` - The device to use.
    pub fn new(embed_dim: usize, num_features: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(embed_dim, num_features / 2).init(device),
        }
    }

    /// Forward pass of the SequentialFovNetworkEncoder module.
    ///
    /// # Arguments
    /// * `input` - The input tensor.
    /// * `device` - The device to use.
    /// * `encoder` - The CommonVitModel encoder.
    pub fn forward<F: MixedFloats>(
        &mut self,
        input: Tensor<B, 4>,
        device: &B::Device,
        encoder: &mut CommonVitModel,
    ) -> Result<Tensor<B, 3>> {
        let encoded = encoder.forward::<B, F>(input, device)?;

        let output = self.linear.forward(encoded.tensor);

        Ok(output)
    }
}
