use anyhow::{Result, anyhow};
use burn::Tensor;
use burn::nn::{BatchNorm, Relu, modules::conv::Conv2d};
use burn::prelude::Backend;

pub struct SequentialNNModule<B: Backend> {
    relu: Relu,
    conv: Conv2d<B>,
    batch_norm: Option<BatchNorm<B>>,
}

impl<B: Backend> SequentialNNModule<B> {
    fn forward(&self, arg: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.relu.forward(arg);
        let x = self.conv.forward(x);

        match &self.batch_norm {
            Some(batch_norm) => batch_norm.forward(x),
            None => x,
        }
    }
}

pub struct ResidualBlock<B: Backend> {
    sequential: SequentialNNModule<B>,
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

        let module = SequentialNNModule {
            relu: Relu::new(),
            conv: conv_config,
            batch_norm: None,
        };

        let res = module.forward(tensor).to_data();
        let values: Vec<f32> = res.to_vec().unwrap();

        assert_eq!(values, [36.5, 36.5, 36.5, 36.5, 35.5, 35.5, 35.5, 35.5]);
    }
}
