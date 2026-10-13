use burn::{
    Tensor,
    module::Module,
    nn::{
        PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Device,
};

/// `SequentialFovNetwork0` is a sequential network that takes an input tensor and applies a convolutional layer followed by a `ReLU` activation.
///
/// This follows the implementation of depth-pro's fov.py. Reference can be found at the link below
///
/// @link <https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L30>.
#[derive(Debug, Module)]
pub struct SequentialFovNetwork0 {
    pub conv: Conv2d,
    pub relu: Relu,
}

/// `SequentialFovNetwork` is a sequential network that takes an input tensor and applies a convolutional layer followed by a `ReLU` activation.
///
/// This follows the implementation of depth-pro's fov.py. Reference can be found at the link below
///
/// @link <https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L36>.
#[derive(Debug, Module)]
pub struct SequentialFovNetwork {
    pub fov_head0: Option<SequentialFovNetwork0>,
    pub conv64: Conv2d,
    pub relu64: Relu,
    pub conv32: Conv2d,
    pub relu32: Relu,
    pub conv16: Conv2d,
}

impl SequentialFovNetwork0 {
    /// Create a new `SequentialFovNetwork0` module.
    ///
    /// # Arguments
    /// * `num_features` - The number of features in the input tensor.
    /// * `device` - The device on which the network will be created.
    pub fn new(num_features: usize, device: &Device) -> Self {
        Self {
            conv: Conv2dConfig::new([num_features, num_features / 2], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            relu: Relu::new(),
        }
    }

    /// Forward pass of the `SequentialFovNetwork0` module.
    ///
    /// # Arguments
    /// * `input` - The input tensor.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        // 128 x 24 x 24
        let output = self.conv.forward(input);

        self.relu.forward(output)
    }
}

impl SequentialFovNetwork {
    /// Create a new `SequentialFovNetwork` instance.
    /// The usage of the `SequentialFovNetwork0` instance is optional and only being used whenever the "`fov_encoder`" is None.
    ///
    /// # Arguments
    /// * `num_features` - The number of features in the input tensor.
    /// * `fov_head0` - An optional `SequentialFovNetwork0` instance.
    /// * `device` - The device on which the network will be created.
    ///
    /// # Returns
    /// A new `SequentialFovNetwork` instance.
    pub fn new(
        num_features: usize,
        fov_head0: Option<SequentialFovNetwork0>,
        device: &Device,
    ) -> Self {
        Self {
            fov_head0,
            conv64: Conv2dConfig::new([num_features / 2, num_features / 4], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            relu64: Relu::new(),
            conv32: Conv2dConfig::new([num_features / 4, num_features / 8], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            relu32: Relu::new(),
            conv16: Conv2dConfig::new([num_features / 8, 1], [6, 6])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
                .init(device),
        }
    }

    /// This implements the forward pass of the sequential network based on a given input tensor
    /// This implementation refers to depth-pro's fov.py. Reference can be found at the link below
    ///
    /// @link <https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L30>.
    ///
    /// # Arguments
    /// * `input` - The input tensor to the network
    ///
    /// # Returns
    /// The output tensor of the network
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
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
    use burn::tensor::{Distribution, Shape};

    use super::*;

    #[test]
    fn expect_basic_forward_pass_to_work() {
        let device = Default::default();
        let num_features = 256;

        let tensor = Tensor::<4>::random(
            Shape::new([1, num_features, 48, 48]),
            Distribution::Uniform(0., 1.),
            &device,
        );

        let fov_head0 = SequentialFovNetwork0::new(num_features, &device);
        let fov_head = SequentialFovNetwork::new(num_features, Some(fov_head0), &device);
        let output = fov_head.forward(tensor);

        assert_eq!(output.shape(), Shape::new([1, 1, 1, 1]));
    }
}
