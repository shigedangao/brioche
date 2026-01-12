use super::residual_block::{ResidualBlock, SequentialNNModule};
use super::{Decoder, DecoderType};
use crate::network::decoder::DecoderOutput;
use anyhow::{Result, anyhow};
use burn::module::Module;
use burn::nn::{
    PaddingConfig2d, Relu,
    conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
};
use burn::prelude::Backend;

/// Feature fusion block for 2D images.
///
/// This implements the FeatureFusionBlock2d class from the deco
/// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/decoder.py#L121
#[derive(Debug, Module)]
pub struct FeatureFusionBlock2D<B: Backend> {
    num_features: usize,
    deconv: Option<ConvTranspose2d<B>>,
    outconv: Conv2d<B>,
    batch_norm: bool,
    resnet1: ResidualBlock<B>,
    resnet2: ResidualBlock<B>,
}

impl<B: Backend> FeatureFusionBlock2D<B> {
    pub fn new(num_features: usize, deconv: bool, batch_norm: bool, device: &B::Device) -> Self {
        let conv_config = Conv2dConfig::new([num_features, num_features], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(true)
            .init::<B>(&device);

        let resnet1 = ResidualBlock::new([
            SequentialNNModule {
                relu: Relu::new(),
                conv2d: conv_config.clone(),
                batch_norm: None,
            },
            SequentialNNModule {
                relu: Relu::new(),
                conv2d: conv_config.clone(),
                batch_norm: None,
            },
        ]);

        let resnet2 = ResidualBlock::new([
            SequentialNNModule {
                relu: Relu::new(),
                conv2d: conv_config.clone(),
                batch_norm: None,
            },
            SequentialNNModule {
                relu: Relu::new(),
                conv2d: conv_config.clone(),
                batch_norm: None,
            },
        ]);

        // 1x1 convolution refines the fused features without changing spatial size
        // Acts as a learned linear transformation to combine feature channels
        let outconv = Conv2dConfig::new([num_features, num_features], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(true)
            .init::<B>(&device);

        let mut ffb2d = Self {
            num_features,
            batch_norm,
            deconv: None,
            outconv,
            resnet1,
            resnet2,
        };

        if deconv {
            // Transpose convolution (deconv) upsamples spatial dimensions by 2x
            // Used to progressively increase resolution when fusing features
            ffb2d.deconv = Some(
                ConvTranspose2dConfig::new([num_features, num_features], [2, 2])
                    .with_stride([2, 2])
                    .with_padding([0, 0])
                    .with_bias(false)
                    .init::<B>(&device),
            );
        }

        ffb2d
    }
}

impl<B: Backend> Decoder<B, 4> for FeatureFusionBlock2D<B> {
    fn forward(&self, arg: DecoderType<B>) -> Result<DecoderOutput<B>> {
        let (mut x0, x1) = match arg {
            DecoderType::FeatureFusionBlock2D(x, x1) => (x, x1),
            _ => return Err(anyhow!("Invalid input type")),
        };

        if let Some(x1) = x1 {
            let (res, _) = self.resnet1.forward(DecoderType::ResidualBlock(x1))?;
            x0 = x0 + res;
        }

        let (mut x0, _) = self.resnet2.forward(DecoderType::ResidualBlock(x0))?;

        if let Some(deconv) = &self.deconv {
            x0 = deconv.forward(x0);
        }

        let out = self.outconv.forward(x0);

        Ok((out, None))
    }
}
