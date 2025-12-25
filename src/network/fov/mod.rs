use super::vit::VitModule;
use burn::{
    Tensor,
    nn::interpolate::{Interpolate2dConfig, InterpolateMode},
    prelude::Backend,
};
use encoder_seq::SequentialFovNetworkEncoder;
use sequential::{SequentialFovNetwork, SequentialFovNetwork0};

mod encoder_seq;
mod sequential;

/// FovNetwork is used to perform some task on the field of view (i guess ?)
/// The implementation details is based on the fov network of depth-pro fov init method. Please follow the link below
///
/// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L14
#[derive(Debug, Clone)]
struct FovNetwork<B: Backend> {
    head: SequentialFovNetwork<B>,
    encoder: Option<SequentialFovNetworkEncoder<B>>,
    downsample: Option<SequentialFovNetwork0<B>>,
}

impl<B: Backend> FovNetwork<B> {
    pub fn new(num_features: usize, fov_encoder: Option<VitModule<B>>, device: B::Device) -> Self {
        match fov_encoder {
            Some(module) => {
                let embed_dim = module.embeded_dim;

                let encoder =
                    SequentialFovNetworkEncoder::new(module, embed_dim, num_features, &device);

                let downsample = SequentialFovNetwork0::new(num_features, &device);
                let head = SequentialFovNetwork::new(num_features, None, &device);

                Self {
                    head,
                    encoder: Some(encoder),
                    downsample: Some(downsample),
                }
            }
            None => {
                let fov_head0 = SequentialFovNetwork0::new(num_features, &device);
                let head = SequentialFovNetwork::new(num_features, Some(fov_head0), &device);

                Self {
                    head,
                    encoder: None,
                    downsample: None,
                }
            }
        }
    }

    fn forward(&self, x: Tensor<B, 4>, lowres_feature: Tensor<B, 4>) -> Tensor<B, 4> {
        match self.encoder {
            Some(ref encoder) => {
                let interpolate = Interpolate2dConfig::new()
                    .with_scale_factor(Some([0.25, 0.25]))
                    .with_mode(InterpolateMode::Linear)
                    .init();

                let encoder_out = interpolate.forward(x);

                // For [:, 1:] slicing - get the shape first
                let [batch, seq_len, hidden, _] = encoder_out.dims();

                // Slice to remove first token (dimension 1, from index 1 onwards)
                let sliced_out = encoder_out.slice([
                    0..batch,   // All batches
                    1..seq_len, // From index 1 to end
                    0..hidden,  // All hidden dimensions
                ]);

                let permuted = sliced_out.swap_dims(1, 2);
                let downsampled_x = match self.downsample {
                    Some(ref downsample) => {
                        let lowres_output = downsample.forward(lowres_feature);

                        permuted + lowres_output
                    }
                    None => permuted + lowres_feature,
                };

                self.head.forward(downsampled_x)
            }
            None => self.head.forward(lowres_feature),
        }
    }
}
