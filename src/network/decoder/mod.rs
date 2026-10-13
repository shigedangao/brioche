use anyhow::Result;
use burn::tensor::Tensor;

mod feature_fusion_block_2d;
pub(crate) mod multires_conv;
mod residual_block;

/// `DecoderType` represents the type of decoder to use.
pub enum DecoderType {
    FeatureFusionBlock2D(Tensor<4>, Box<Option<Tensor<4>>>),
    MultiResConv(Vec<Tensor<4>>),
    ResidualBlock(Tensor<4>),
}

/// `DecoderOutput` represents the output of the decoder.
pub type DecoderOutput = (Tensor<4>, Option<Tensor<4>>);

/// `Decoder` define the trait for the decoder to be use by any type of decoder.
pub trait Decoder<const S: usize> {
    /// Forward pass of the decoder.
    ///
    /// # Arguments
    ///
    /// * `arg` - The input to the decoder.
    fn forward(&self, arg: DecoderType) -> Result<DecoderOutput>;
}
