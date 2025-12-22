use anyhow::Result;
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorKind};

mod feature_fusion_block_2d;
mod residual_block;

pub enum DecoderType<B: Backend> {
    FeatureFusionBlock2D(Tensor<B, 4>, Option<Tensor<B, 4>>),
    ResidualBlock(Tensor<B, 4>),
}

pub trait Decoder<B: Backend, const S: usize> {
    fn forward(&self, arg: DecoderType<B>) -> Result<Tensor<B, S>>;
}
