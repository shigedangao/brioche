use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorKind};

mod residual_block;

pub enum DecoderType {
    MultiresConv,
    ResidualBlock,
}

pub trait Decoder {
    fn forward<B: Backend, const S: usize>(
        &self,
        arg: DecoderType,
    ) -> Result<Tensor<B, S>, Box<dyn std::error::Error>>;
}
