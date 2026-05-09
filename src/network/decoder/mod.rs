use anyhow::Result;
use burn::prelude::Backend;
use burn::tensor::Tensor;

mod feature_fusion_block_2d;
pub(crate) mod multires_conv;
mod residual_block;

/// `DecoderType` represents the type of decoder to use.
pub enum DecoderType<B: Backend> {
    FeatureFusionBlock2D(Tensor<B, 4>, Option<Tensor<B, 4>>),
    MultiResConv(Vec<Tensor<B, 4>>),
    ResidualBlock(Tensor<B, 4>),
}

/// `DecoderOutput` represents the output of the decoder.
pub type DecoderOutput<B> = (Tensor<B, 4>, Option<Tensor<B, 4>>);

/// `Decoder` define the trait for the decoder to be use by any type of decoder.
pub trait Decoder<B: Backend, const S: usize> {
    /// Forward pass of the decoder.
    ///
    /// # Arguments
    ///
    /// * `arg` - The input to the decoder.
    fn forward(&self, arg: DecoderType<B>) -> Result<DecoderOutput<B>>;
}
