use burn::{
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Debug, Module)]
pub struct ProjectionSeq<B: Backend> {
    // Conv2d use for the projection
    conv2d: Conv2d<B>,
    // ConvTranspose2d use for the upsampling
    blocks: Vec<ConvTranspose2d<B>>,
}

impl<B: Backend> ProjectionSeq<B> {
    /// Create a new ProjectionSeq module.
    ///
    /// # Arguments
    /// * `dims_in` - The input dimension.
    /// * `dims_int` - The intermediate dimension.
    /// * `dims_out` - The output dimension.
    /// * `upsample_layer` - The number of upsampling layers.
    /// * `device` - The device to use.
    pub fn new(
        dims_in: usize,
        dims_int: Option<usize>,
        dims_out: usize,
        upsample_layer: usize,
        device: &B::Device,
    ) -> Self {
        // Upsampling
        let mut blocks = Vec::new();

        let conv2d_dims_out = match dims_int {
            Some(dims_int) => dims_int,
            None => dims_out,
        };

        for i in 0..upsample_layer {
            let channels = match i {
                0 => [conv2d_dims_out, dims_out],
                _ => [dims_out, dims_out],
            };

            blocks.push(
                ConvTranspose2dConfig::new(channels, [2, 2])
                    .with_stride([2, 2])
                    .with_padding([0, 0])
                    .with_bias(false)
                    .init::<B>(device),
            );
        }

        Self {
            // Projection block.
            conv2d: Conv2dConfig::new([dims_in, conv2d_dims_out], [1, 1])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0))
                .with_bias(false)
                .init::<B>(device),
            blocks,
        }
    }

    /// Forward pass of the ProjectionSeq module.
    ///
    /// 1. Apply the initial Conv2d layer to the input tensor.
    /// 2. Apply each ConvTranspose2d layer in sequence to the output tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor to the ProjectionSeq module.
    pub fn forward(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut output = self.conv2d.forward(tensor);
        for block in &self.blocks {
            output = block.forward(output);
        }

        output
    }
}
