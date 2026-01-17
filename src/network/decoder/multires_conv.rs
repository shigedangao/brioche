use super::feature_fusion_block_2d::FeatureFusionBlock2D;
use super::{Decoder, DecoderType};
use crate::network::{Network, NetworkConfig, decoder::DecoderOutput};
use anyhow::{Result, anyhow};
use burn::module::Module;
use burn::{
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::Backend,
};

#[derive(Debug, Module)]
pub struct MultiResConv<B: Backend> {
    dims_encoder_len: usize,
    convs: Vec<Conv2d<B>>,
    fusions: Vec<FeatureFusionBlock2D<B>>,
}

#[derive(Debug, Default)]
pub struct MultiResDecoderConfig {
    pub dims_encoder: Vec<usize>,
    pub dim_decoder: usize,
}

impl<B: Backend> Network<B> for MultiResConv<B> {
    fn new(config: NetworkConfig, device: &B::Device) -> Result<Self> {
        let NetworkConfig::Decoder(config) = config else {
            return Err(anyhow!("Invalid network configuration"));
        };

        let MultiResDecoderConfig {
            dims_encoder,
            dim_decoder,
        } = config;

        let conv0 = dims_encoder.first().and_then(|dim| match dim {
            v if *v == dim_decoder => None,
            _ => {
                let conv_config = Conv2dConfig::new([*dim, dim_decoder], [1, 1])
                    .with_stride([1, 1])
                    .with_padding(PaddingConfig2d::Explicit(1, 1))
                    .with_bias(false)
                    .init::<B>(&device);

                Some(conv_config)
            }
        });

        // @TODO maybe push Option<Conv2d<B>> in order to match the same weight import as depth-pro
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
            .map(|(idx, _)| FeatureFusionBlock2D::new(dim_decoder, idx != 0, device))
            .collect();

        Ok(Self {
            dims_encoder_len: dims_encoder.len(),
            convs,
            fusions,
        })
    }
}

impl<B: Backend> Decoder<B, 4> for MultiResConv<B> {
    fn forward(&self, input: DecoderType<B>) -> Result<DecoderOutput<B>> {
        let encodings = match input {
            DecoderType::MultiResConv(arg) => arg,
            _ => return Err(anyhow!("Invalid input type")),
        };

        if encodings.len() != self.dims_encoder_len {
            return Err(anyhow!(
                "Got encoder output levels={}, expected levels={}.",
                encodings.len(),
                self.dims_encoder_len
            ));
        }

        // /!\ Reprocess the self.convs.
        //
        // If the encodings len is different than the conv itself
        // it means that the first element is empty. Due to the fact that burn's weight import
        // required the convs to be the same length. We could not have an array length different than the pt that's being imported.
        // When the dim_decoder = dim_encoder[0] the depth-pro library use the nn.Identity module (which doesn't exist in burn yet).
        // As a result we'll reprocess the self.convs and transform them as optional with None as the first item if the length is different than the encodings length.
        let mut reprocessed_convs = Vec::with_capacity(encodings.len());
        if encodings.len() != self.convs.len() {
            // Push the first element as None
            reprocessed_convs.push(None);

            for conv in self.convs.iter() {
                reprocessed_convs.push(Some(conv));
            }
        }

        // Project features of different encoder dims to the same decoder dim.
        // Fuse features from the lowest resolution (num_levels-1)
        // to the highest (0).
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

        let last_fusion = self
            .fusions
            .last()
            .ok_or(anyhow!("Unable to get the latest fusion"))?;

        let (last_fusion_output, _) =
            last_fusion.forward(DecoderType::FeatureFusionBlock2D(features.clone(), None))?;

        features = last_fusion_output;

        for idx in (0..encodings.len() - 1).rev() {
            let conv = reprocessed_convs
                .get(idx)
                .ok_or(anyhow!("Unable to get tensor at index {idx}"))?;

            let encoding = encodings
                .get(idx)
                .ok_or(anyhow!("Unable to get encoding at index {idx}"))?;

            let features_idx = match conv {
                Some(cnv) => cnv.forward(encoding.clone()),
                // Passthrough
                None => encoding.clone(),
            };

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

#[cfg(test)]
mod tests {
    use crate::{Decoder, DecoderType};
    use burn::backend::Metal;
    use burn::tensor::{Tensor, TensorData};
    use ndarray::Array4;

    use crate::network::{
        Network, NetworkConfig,
        decoder::multires_conv::{MultiResConv, MultiResDecoderConfig},
    };

    fn create_decoder() -> MultiResConv<Metal> {
        let device = Default::default();

        let decoder = MultiResConv::new(
            NetworkConfig::Decoder(MultiResDecoderConfig {
                dims_encoder: vec![256, 256, 512, 1024, 1024],
                dim_decoder: 256,
            }),
            &device,
        )
        .unwrap()
        .with_record(
            "/Users/marcintha/workspace/brioche/butter/decoder_only.pt",
            &device,
        );

        decoder
    }

    #[test]
    fn expect_decoder_to_output_something() {
        let device = Default::default();

        // x_latent0_features
        let x_latent0_features_np: Array4<f32> =
            ndarray_npy::read_npy("testdata/tensors_data/decoder/encodings_x_latent0_features.npy")
                .unwrap();
        let (x_latent0_data, _) = x_latent0_features_np.into_raw_vec_and_offset();
        let x_latent0_features: Tensor<Metal, 4> =
            Tensor::from_data(TensorData::new(x_latent0_data, [1, 256, 768, 768]), &device);

        // x_latent1_features
        let x_latent1_features_np: Array4<f32> =
            ndarray_npy::read_npy("testdata/tensors_data/decoder/encodings_x_latent1_features.npy")
                .unwrap();
        let (x_latent1_data, _) = x_latent1_features_np.into_raw_vec_and_offset();
        let x_latent1_features: Tensor<Metal, 4> =
            Tensor::from_data(TensorData::new(x_latent1_data, [1, 256, 384, 384]), &device);

        // x0_features
        let x0_features_np: Array4<f32> =
            ndarray_npy::read_npy("testdata/tensors_data/decoder/encodings_x0_features.npy")
                .unwrap();
        let (x0_features_data, _) = x0_features_np.into_raw_vec_and_offset();
        let x0_features: Tensor<Metal, 4> = Tensor::from_data(
            TensorData::new(x0_features_data, [1, 512, 192, 192]),
            &device,
        );

        // x1_features
        let x1_features_np: Array4<f32> =
            ndarray_npy::read_npy("testdata/tensors_data/decoder/encodings_x1_features.npy")
                .unwrap();
        let (x1_features_data, _) = x1_features_np.into_raw_vec_and_offset();
        let x1_features: Tensor<Metal, 4> = Tensor::from_data(
            TensorData::new(x1_features_data, [1, 1024, 96, 96]),
            &device,
        );

        // x_global_features
        let x_global_features_np: Array4<f32> =
            ndarray_npy::read_npy("testdata/tensors_data/decoder/encodings_x_global_features.npy")
                .unwrap();
        let (x0_global_data, _) = x_global_features_np.into_raw_vec_and_offset();
        let x_global_features: Tensor<Metal, 4> =
            Tensor::from_data(TensorData::new(x0_global_data, [1, 1024, 48, 48]), &device);

        let decoder = create_decoder();
        let output = decoder.forward(DecoderType::MultiResConv(vec![
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        ]));

        assert!(output.is_ok());
        let res = output.unwrap();

        assert_eq!(res.0.shape().dims(), [1, 256, 768, 768]);
        assert_eq!(res.1.unwrap().shape().dims(), [1, 256, 48, 48]);
    }
}
