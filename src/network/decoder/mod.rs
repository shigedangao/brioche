use anyhow::Result;
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorKind};

mod residual_block;

pub enum DecoderType<B: Backend> {
    MultiresConv,
    ResidualBlock(Tensor<B, 4>),
}

pub trait Decoder<B: Backend, const S: usize> {
    fn forward(&self, arg: DecoderType<B>) -> Result<Tensor<B, S>>;
}
