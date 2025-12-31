use crate::vit::fov::FovVitModel;
use anyhow::{Result, anyhow};
use burn::{
    Tensor,
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::Backend,
};
use ndarray::Array4;

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
    pub fn new(embed_dim: usize, num_features: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(embed_dim, num_features / 2).init(device),
        }
    }

    pub fn forward(
        &mut self,
        input: Tensor<B, 4>,
        device: &B::Device,
        encoder: &mut FovVitModel,
    ) -> Result<Tensor<B, 3>> {
        let array: Vec<f32> = input
            .to_data()
            .to_vec()
            .map_err(|err| anyhow!("Unable to convert the tensor to a vector due to {:?}", err))?;

        let tensor_ndarray = Array4::from_shape_vec((1, 3, 384, 384), array)?;
        let encoded = encoder.forward(tensor_ndarray, device)?;

        let output = self.linear.forward(encoded);

        Ok(output)
    }
}
