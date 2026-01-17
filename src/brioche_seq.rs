use burn::{
    module::Module,
    nn::{
        PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Debug, Module)]
pub struct BriocheSeq<B: Backend> {
    conv2d0: Conv2d<B>,
    conv_transpose2d: ConvTranspose2d<B>,
    conv2d1: Conv2d<B>,
    relu0: Relu,
    conv2d2: Conv2d<B>,
    relu1: Relu,
}

impl<B: Backend> BriocheSeq<B> {
    /// Creates a new instance of `BriocheSeq`.
    /// This refer to the "head" sequential part of the DepthPro module. Refer to the link below
    /// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/depth_pro.py#L182
    ///
    /// # Arguments
    /// * dim_decoder - usize
    /// * last_dims - (usize, usize)
    /// * device: &B::Device
    pub fn new(dim_decoder: usize, last_dims: (usize, usize), device: &B::Device) -> Self {
        let conv2d0 = Conv2dConfig::new([dim_decoder, dim_decoder / 2], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(true)
            .init::<B>(&device);

        let conv_transpose2d =
            ConvTranspose2dConfig::new([dim_decoder / 2, dim_decoder / 2], [2, 2])
                .with_stride([2, 2])
                .with_padding([0, 0])
                .with_bias(false)
                .init::<B>(&device);

        let conv2d1 = Conv2dConfig::new([dim_decoder / 2, last_dims.0], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init::<B>(&device);

        let mut conv2d2 = Conv2dConfig::new([last_dims.0, last_dims.1], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .init::<B>(&device);

        // Set the final convolution layer's bias to be 0.
        if let Some(bias) = &mut conv2d2.bias {
            let res = bias.clone().map(|t| {
                let req = t.is_require_grad();
                // Fresh zeros tensor, same shape/device as the existing bias
                Tensor::<B, 1>::zeros(t.dims(), &t.device()).set_require_grad(req)
            });

            *bias = res;
        }

        Self {
            conv2d0,
            conv_transpose2d,
            conv2d1,
            relu0: Relu::new(),
            conv2d2,
            relu1: Relu::new(),
        }
    }

    /// Forward pass of the network.
    ///
    /// # Arguments
    /// * input - Tensor<B, 4>
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv2d0.forward(input);
        let x = self.conv_transpose2d.forward(x);
        let x = self.conv2d1.forward(x);
        let x = self.relu0.forward(x);
        let x = self.conv2d2.forward(x);
        let x = self.relu1.forward(x);

        x
    }
}
