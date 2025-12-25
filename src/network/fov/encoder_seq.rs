use crate::network::vit::VitModule;
use burn::{
    Tensor,
    nn::{Linear, LinearConfig},
    prelude::Backend,
};

/// Sequential Foveal Vision Transformer (FOV) network encoder.
///
/// This module implements the following fov part of depth-pro. refer to the link below
/// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L48
///
/// /!\ Note that this is tightly linked to the fov_encoder module that needs to be converted with onnx.
#[derive(Debug, Clone)]
pub struct SequentialFovNetworkEncoder<B: Backend> {
    fov_encoder: VitModule<B>,
    linear: Linear<B>,
}

impl<B: Backend> SequentialFovNetworkEncoder<B> {
    pub fn new(
        fov_encoder: VitModule<B>,
        embed_dim: usize,
        num_features: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            fov_encoder,
            linear: LinearConfig::new(embed_dim, num_features / 4).init(device),
        }
    }

    fn forward<const S: usize>(&self, input: Tensor<B, S>) -> Tensor<B, S> {
        let encoded = self.fov_encoder.forward(input);
        let output = self.linear.forward(encoded);

        output
    }
}
