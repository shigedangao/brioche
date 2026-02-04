use super::{Network, NetworkConfig};
use crate::MixedFloats;
use anyhow::{Result, anyhow};
use burn::{Tensor, module::Module, prelude::Backend};
use encoder_seq::SequentialFovNetworkEncoder;
use sequential::{SequentialFovNetwork, SequentialFovNetwork0};

mod encoder_seq;
mod sequential;

/// Fov is used to perform some task on the field of view (i guess ?)
/// The implementation details is based on the fov network of depth-pro fov init method. Please follow the link below
///
/// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L14
#[derive(Debug, Module)]
pub struct Fov<B: Backend> {
    head: SequentialFovNetwork<B>,
    encoder: Option<SequentialFovNetworkEncoder<B>>,
    downsample: Option<SequentialFovNetwork0<B>>,
}

#[derive(Debug, Clone)]
pub struct FovConfig {
    pub num_features: usize,
    pub with_fov_encoder: bool,
    pub embed_dim: usize,
}

impl<B: Backend> Network<B> for Fov<B> {
    /// Create a new FovNetwork instance.
    /// /!\ Note that we could not pass the fov_encoder directly as the Session does not have the Copy & Clone trait.
    ///     As a result, it's not possible to pass the fov_encoder as an Option
    ///
    /// # Arguments
    ///
    /// * `num_features` - The number of features.
    /// * `with_fov_encoder` - Whether to use the fov encoder.
    /// * `device` - The device to use.
    ///
    /// # Returns
    ///
    /// A new FovNetwork instance.
    fn new(config: NetworkConfig, device: &B::Device) -> Result<Self> {
        let NetworkConfig::Fov(config) = config else {
            return Err(anyhow!("Invalid network configuration"));
        };

        let FovConfig {
            num_features,
            with_fov_encoder,
            embed_dim,
        } = config;

        let fov_head0 = SequentialFovNetwork0::new(num_features, device);

        let fov = match with_fov_encoder {
            true => Self {
                head: SequentialFovNetwork::new(num_features, None, device),
                encoder: Some(SequentialFovNetworkEncoder::new(
                    embed_dim,
                    num_features,
                    device,
                )),
                downsample: Some(fov_head0),
            },
            false => Self {
                head: SequentialFovNetwork::new(num_features, Some(fov_head0), device),
                encoder: None,
                downsample: None,
            },
        };

        Ok(fov)
    }
}

impl<B: Backend> Fov<B> {
    /// Forward compute the output of the network.
    ///
    /// # Arguments
    /// * `x` - The input tensor.
    /// * `lowres_feature` - The low resolution feature tensor.
    /// * `device` - The device to use.
    /// * `fov_encoder` - The fov encoder.
    ///
    /// # Returns
    /// The output tensor.
    pub fn forward<F: MixedFloats>(
        &mut self,
        x: Tensor<B, 3>,
        lowres_feature: Tensor<B, 4>,
    ) -> Result<Tensor<B, 4>> {
        let out = match self.encoder {
            Some(ref mut encoder) => {
                // Encode the interpolated features
                let encoder_out = encoder.forward::<F>(x)?;

                // Slice to remove first token (dimension 1, from index 1 onwards)
                let sliced_out = encoder_out.slice([
                    0.., // All batches
                    1.., // From index 1 to end
                    0.., // All hidden dimensions
                ]);

                let permuted = sliced_out.permute([0, 2, 1]);

                let processed_x = match self.downsample {
                    Some(ref downsample) => {
                        let lowres_output = downsample.forward(lowres_feature);
                        let x_tensor4 = permuted.reshape(lowres_output.shape());

                        x_tensor4 + lowres_output
                    }
                    None => lowres_feature,
                };

                self.head.forward(processed_x)
            }
            None => self.head.forward(lowres_feature),
        };

        Ok(out)
    }
}
