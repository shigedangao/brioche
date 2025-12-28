use burn::{
    Tensor,
    nn::{
        PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
};

/// SequentialFovNetwork0 is a sequential network that takes an input tensor and applies a convolutional layer followed by a ReLU activation.
///
/// This follows the implementation of depth-pro's fov.py. Reference can be found at the link below
///
/// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L30.
#[derive(Debug, Clone)]
pub struct SequentialFovNetwork0<B: Backend> {
    pub conv: Conv2d<B>,
    pub relu: Relu,
}

/// SequentialFovNetwork is a sequential network that takes an input tensor and applies a convolutional layer followed by a ReLU activation.
///
/// This follows the implementation of depth-pro's fov.py. Reference can be found at the link below
///
/// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L36.
#[derive(Debug, Clone)]
pub struct SequentialFovNetwork<B: Backend> {
    pub fov_head0: Option<SequentialFovNetwork0<B>>,
    pub conv64: Conv2d<B>,
    pub relu64: Relu,
    pub conv32: Conv2d<B>,
    pub relu32: Relu,
    pub conv16: Conv2d<B>,
}

impl<B: Backend> SequentialFovNetwork0<B> {
    pub fn new(num_features: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv2dConfig::new(
                [num_features, (num_features as f64 / 2.).floor() as usize],
                [3, 3],
            )
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init::<B>(&device),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // 128 x 24 x 24
        let output = self.conv.forward(input);
        let output = self.relu.forward(output);

        output
    }
}

impl<B: Backend> SequentialFovNetwork<B> {
    /// Create a new SequentialFovNetwork instance.
    /// The usage of the SequentialFovNetwork0 instance is optional and only being used whenever the "fov_encoder" is None.
    ///
    /// # Arguments
    /// * `num_features` - The number of features in the input tensor.
    /// * `fov_head0` - An optional SequentialFovNetwork0 instance.
    /// * `device` - The device on which the network will be created.
    ///
    /// # Returns
    /// A new SequentialFovNetwork instance.
    pub fn new(
        num_features: usize,
        fov_head0: Option<SequentialFovNetwork0<B>>,
        device: &B::Device,
    ) -> Self {
        Self {
            fov_head0,
            conv64: Conv2dConfig::new([num_features / 2, num_features / 4], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init::<B>(&device),
            relu64: Relu::new(),
            conv32: Conv2dConfig::new([num_features / 4, num_features / 8], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init::<B>(&device),
            relu32: Relu::new(),
            conv16: Conv2dConfig::new([num_features / 8, 1], [6, 6])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Valid)
                .init::<B>(&device),
        }
    }

    /// This implements the forward pass of the sequential network based on a given input tensor
    /// This implementation refers to depth-pro's fov.py. Reference can be found at the link below
    ///
    /// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L30.
    ///
    /// # Arguments
    /// * `input` - The input tensor to the network
    ///
    /// # Returns
    /// The output tensor of the network
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let fov0_output = match self.fov_head0 {
            Some(ref fov_head0) => fov_head0.forward(input),
            None => input,
        };

        // 64 x 12 x 12
        let mut output = self.conv64.forward(fov0_output);
        output = self.relu64.forward(output);

        // 32 x 6 x 6
        output = self.conv32.forward(output);
        output = self.relu32.forward(output);

        // 16 x 3 x 3
        output = self.conv16.forward(output);

        output
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Wgpu,
        tensor::{Distribution, Shape},
    };

    use super::*;

    #[test]
    fn expect_basic_forward_pass_to_work() {
        let device = Default::default();
        let num_features = 256;

        let tensor = Tensor::<Wgpu, 4>::random(
            Shape::new([1, num_features, 48, 48]),
            Distribution::Uniform(0., 1.),
            &device,
        );

        let fov_head0 = SequentialFovNetwork0::<Wgpu>::new(num_features, &device);

        let fov_head = SequentialFovNetwork::<Wgpu>::new(num_features, Some(fov_head0), &device);

        let output = fov_head.forward(tensor);

        assert_eq!(output.shape(), Shape::new([1, 1, 1, 1]));
    }
}
