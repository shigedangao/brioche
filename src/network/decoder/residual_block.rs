use super::{Decoder, DecoderOutput, DecoderType};
use anyhow::{Result, anyhow};
use burn::Tensor;
use burn::module::Module;
use burn::nn::{BatchNorm, Relu, modules::conv::Conv2d};
use burn::prelude::Backend;

/// Sequential neural network module
///
/// This implements the _create_block method in python.
/// for reference please see: depth_pro/network/decoder.py L:191
#[derive(Debug, Module)]
pub struct SequentialNNModule<B: Backend> {
    pub relu: Relu,
    pub conv2d: Conv2d<B>,
    pub batch_norm: Option<BatchNorm<B>>,
}

impl<B: Backend> SequentialNNModule<B> {
    fn forward(&self, arg: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.relu.forward(arg);
        let x = self.conv2d.forward(x);

        match &self.batch_norm {
            Some(batch_norm) => batch_norm.forward(x),
            None => x,
        }
    }
}

/// Residual block module
///
/// Based on the implementation in depth_pro/network/decoder.py L:96
/// /!\ Note that the shortcut is not implemented as it seems unused.
#[derive(Debug, Module)]
pub struct ResidualBlock<B: Backend> {
    sequential: [SequentialNNModule<B>; 2],
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new(sequential: [SequentialNNModule<B>; 2]) -> Self {
        Self { sequential }
    }
}

impl<B: Backend> Decoder<B, 4> for ResidualBlock<B> {
    fn forward(&self, arg: DecoderType<B>) -> Result<DecoderOutput<B>> {
        let tensor = match arg {
            DecoderType::ResidualBlock(tensor) => tensor,
            _ => return Err(anyhow!("Invalid decoder type")),
        };

        let forward_init = self
            .sequential
            .first()
            .map(|seq| seq.forward(tensor))
            .ok_or(anyhow!(
                "Expect to compute the tensor for the first sequential nn"
            ))?;

        let delta_x = self
            .sequential
            .last()
            .map(|seq| seq.forward(forward_init))
            .ok_or(anyhow!(
                "Expect to compute the tensor for the second sequential nn"
            ))?;

        Ok((delta_x, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        Tensor,
        backend::Wgpu,
        module::Param,
        nn::{PaddingConfig2d, conv::Conv2dConfig},
    };

    #[test]
    fn test_sequential_nn_module() {
        let device = Default::default();
        // Create a tensor with the following shape: [batch_size, channels, height, width]
        // in a cube
        //  - the channels is the value Z
        //  - the height is the value Y
        //  - the width is the value X
        //
        // So in this example we have a cube with dimensions 2x2x2
        // [
        //  └─ Sample 0 (batch dimension)
        //     [
        //      ├─ Channel 0
        //      │   [
        //      │    [ 1.,  2. ]   ← row 0 (height)
        //      │    [ 3.,  4. ]   ← row 1 (height)
        //      │   ]
        //      │      ↑    ↑
        //      │      w0   w1     (width)
        //      │
        //      ├─ Channel 1
        //      │   [
        //      │    [ 5.,  6. ]   ← row 0 (height)
        //      │    [ 7.,  8. ]   ← row 1 (height)
        //      │   ]
        //      │      ↑    ↑
        //      │      w0   w1     (width)
        //      │
        //     ]
        // ]
        let tensor =
            Tensor::<Wgpu, 4>::from_data([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]], &device);

        // This is based on the following python code
        //
        // nn.Conv2d(
        //        2,
        //        2,
        //        kernel_size=3,
        //        stride=1,
        //        padding=1,
        //        bias=True,
        //    ),

        // Channels -> [input_channels, output_channels]
        // kernel_size - [height, width] (in pytorch you can set 1 value. The other will be set to the same value)
        let mut conv_config = Conv2dConfig::new([2, 2], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(true)
            .init::<Wgpu>(&device);

        // The weight tensor shape is [output_channels, input_channels, kernel_height, kernel_width]
        conv_config.weight = Param::from_tensor(Tensor::full([2, 2, 3, 3], 1., &device));
        conv_config.bias = Some(Param::from_tensor(Tensor::<Wgpu, 1>::from_data(
            [0.5, -0.5],
            &device,
        )));

        let residual_block = ResidualBlock {
            sequential: [
                SequentialNNModule {
                    // Relu -> Transform the negative values to zero
                    relu: Relu::new(),
                    // Conv2d -> extracts spatial features using learned filters (slice over image, detect patterns)
                    conv2d: conv_config.clone(),
                    batch_norm: None,
                },
                SequentialNNModule {
                    relu: Relu::new(),
                    conv2d: conv_config.clone(),
                    batch_norm: None,
                },
            ],
        };

        let computed_tensor_two = residual_block
            .forward(DecoderType::ResidualBlock(tensor))
            .expect("Expect to have compute the residual block tensor")
            .0
            .to_data();
        let nn_sequential_second_layers: Vec<f32> = computed_tensor_two.to_vec().unwrap();

        assert_eq!(
            nn_sequential_second_layers,
            [288.5, 288.5, 288.5, 288.5, 287.5, 287.5, 287.5, 287.5]
        );
    }
}
