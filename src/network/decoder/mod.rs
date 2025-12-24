use anyhow::Result;
use burn::prelude::Backend;
use burn::tensor::Tensor;

mod feature_fusion_block_2d;
mod multires_conv;
mod residual_block;

pub enum DecoderType<B: Backend> {
    FeatureFusionBlock2D(Tensor<B, 4>, Option<Tensor<B, 4>>),
    MultiResConv(Vec<Tensor<B, 4>>),
    ResidualBlock(Tensor<B, 4>),
}

pub type DecoderOutput<B: Backend> = (Tensor<B, 4>, Option<Tensor<B, 4>>);

pub trait Decoder<B: Backend, const S: usize> {
    fn forward(&self, arg: DecoderType<B>) -> Result<DecoderOutput<B>>;
}
