use super::feature_fusion_block_2d::FeatureFusionBlock2D;
use super::{Decoder, DecoderType};
use crate::network::decoder::DecoderOutput;
use anyhow::{Result, anyhow};
use burn::{
    Tensor,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
};

#[derive(Debug, Clone)]
pub struct MultiResConv<B: Backend> {
    dims_encoder: Vec<usize>,
    convs: Vec<Conv2d<B>>,
    fusions: Vec<FeatureFusionBlock2D<B>>,
}

impl<B: Backend> MultiResConv<B> {
    fn new(dims_encoder: Vec<usize>, dim_decoder: usize) -> Self {
        let device = Default::default();

        let conv0 = dims_encoder.first().and_then(|dim| match dim {
            v if *v == dim_decoder => None,
            _ => {
                let conv_config = Conv2dConfig::new([*dim, dim_decoder], [3, 3])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .with_bias(false)
                    .init::<B>(&device);

                Some(conv_config)
            }
        });

        let mut convs = match conv0 {
            Some(conv) => vec![conv],
            None => vec![],
        };

        dims_encoder.iter().skip(1).for_each(|dim| {
            let conv_config = Conv2dConfig::new([*dim, dim_decoder], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .with_bias(false)
                .init::<B>(&device);

            convs.push(conv_config);
        });

        let fusions: Vec<FeatureFusionBlock2D<B>> = dims_encoder
            .iter()
            .enumerate()
            .map(|(idx, _)| FeatureFusionBlock2D::new(dim_decoder, idx != 0, false))
            .collect();

        Self {
            dims_encoder,
            convs,
            fusions,
        }
    }
}

impl<B: Backend> Decoder<B, 4> for MultiResConv<B> {
    fn forward(&self, input: DecoderType<B>) -> Result<DecoderOutput<B>> {
        let encodings = match input {
            DecoderType::MultiResConv(arg) => arg,
            _ => return Err(anyhow!("Invalid input type")),
        };

        if encodings.len() != self.dims_encoder.len() {
            return Err(anyhow!(
                "Got encoder output levels={}, expected levels={}.",
                encodings.len(),
                self.dims_encoder.len()
            ));
        }

        let mut features = self
            .convs
            .last()
            .map(|conv| {
                (
                    conv,
                    encodings
                        .last()
                        .expect("Expect to get the last tensor available"),
                )
            })
            .map(|(conv, encoding)| conv.forward(encoding.clone()))
            .ok_or(anyhow!("Failed to forward the last convolution"))?;

        // Make a copy of the low-resolution features
        let low_res_features = features.clone();

        for idx in (0..encodings.len() - 1).rev() {
            let conv = self
                .convs
                .get(idx)
                .ok_or(anyhow!("Unable to get tensor at index {idx}"))?;

            let encoding = encodings
                .get(idx)
                .ok_or(anyhow!("Unable to get encoding at index {idx}"))?;

            let features_idx = conv.forward(encoding.clone());

            let fusion = self
                .fusions
                .get(idx)
                .ok_or(anyhow!("Unable to get the fusion block"))?;

            let (feat, _) = fusion.forward(DecoderType::FeatureFusionBlock2D(
                features,
                Some(features_idx),
            ))?;

            features = feat;
        }

        Ok((features, Some(low_res_features)))
    }
}
