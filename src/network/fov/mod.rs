use crate::vit::common::CommonVitModel;
use anyhow::Result;
use burn::{
    Tensor,
    module::Module,
    nn::interpolate::{Interpolate2dConfig, InterpolateMode},
    prelude::Backend,
};
use encoder_seq::SequentialFovNetworkEncoder;
use sequential::{SequentialFovNetwork, SequentialFovNetwork0};

mod encoder_seq;
mod sequential;

/// Fov is used to perform some task on the field of view (i guess ?)
/// The implementation details is based on the fov network of depth-pro fov init method. Please follow the link below
///
/// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/fov.py#L14
#[derive(Debug, Module)]
struct Fov<B: Backend> {
    head: SequentialFovNetwork<B>,
    encoder: Option<SequentialFovNetworkEncoder<B>>,
    downsample: Option<SequentialFovNetwork0<B>>,
}

impl<B: Backend> Fov<B> {
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
    pub fn new(
        num_features: usize,
        with_fov_encoder: bool,
        embed_dim: usize,
        device: &B::Device,
    ) -> Self {
        let fov_head0 = SequentialFovNetwork0::new(num_features, device);

        match with_fov_encoder {
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
        }
    }

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
    pub fn forward(
        &mut self,
        x: Tensor<B, 4>,
        lowres_feature: Tensor<B, 4>,
        device: &B::Device,
        fov_encoder: Option<CommonVitModel>,
    ) -> Result<Tensor<B, 4>> {
        let out = match self.encoder {
            Some(ref mut encoder) => {
                let interpolate = Interpolate2dConfig::new()
                    .with_scale_factor(Some([0.25, 0.25]))
                    .with_output_size(None)
                    .with_mode(InterpolateMode::Linear)
                    .init();

                let interpolated_out = interpolate.forward(x);

                // Encode the interpolated features
                let mut fov_encoder = fov_encoder.expect("Expect fov encoder to be present");
                let encoder_out = encoder.forward(interpolated_out, &device, &mut fov_encoder)?;

                // For [:, 1:] slicing - get the shape first
                let [batch, seq_len, hidden] = encoder_out.dims();

                // Slice to remove first token (dimension 1, from index 1 onwards)
                let sliced_out = encoder_out.slice([
                    0..batch,   // All batches
                    1..seq_len, // From index 1 to end
                    0..hidden,  // All hidden dimensions
                ]);

                let permuted = sliced_out.swap_dims(1, 2);

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::record::{FullPrecisionSettings, Recorder};
    use burn::{
        backend::Metal,
        tensor::{Distribution, Shape, TensorData},
    };
    use burn_import::pytorch::PyTorchFileRecorder;
    use ndarray::Array4;
    use std::path::PathBuf;

    fn create_fov_model_with_weight() -> Fov<Metal> {
        let device = Default::default();

        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(
                "/Users/marcintha/workspace/brioche/butter/fov_only.pt".into(),
                &device,
            )
            .unwrap();

        let fov = Fov::<Metal>::new(256, true, 1024, &device).load_record(record);

        fov
    }

    #[test]
    fn expect_fov_test_to_output_something() {
        let device = Default::default();

        let fov_encoder = CommonVitModel::new(
            PathBuf::from("/Users/marcintha/workspace/brioche/butter/depthpro_vit_fov.onnx"),
            4,
        );
        assert!(fov_encoder.is_ok());

        let fov_encoder = fov_encoder.unwrap();

        let mut fov = Fov::<Metal>::new(256, true, 1024, &device);

        let x: Tensor<Metal, 4> = Tensor::random(
            Shape::new([1, 3, 1536, 1536]),
            Distribution::Uniform(-1., 1.),
            &device,
        );

        let lowres_feature: Tensor<Metal, 4> = Tensor::random(
            Shape::new([1, 256, 48, 48]),
            Distribution::Uniform(-6.5, 7.1),
            &device,
        );

        let res = fov.forward(x, lowres_feature, &device, Some(fov_encoder));
        assert!(res.is_ok());
    }

    #[test]
    fn expect_to_run_with_deterministic_tensors() {
        let device = Default::default();

        // Create x
        let x_matrix: Array4<f32> =
            ndarray_npy::read_npy("testdata/tensors_data/fov/x.npy").unwrap();
        let (x_data, _) = x_matrix.into_raw_vec_and_offset();
        let x: Tensor<Metal, 4> =
            Tensor::from_data(TensorData::new(x_data, [1, 3, 1536, 1536]), &device);

        // Create low_res
        let low_res: Array4<f32> =
            ndarray_npy::read_npy("testdata/tensors_data/fov/lowres_feature.npy").unwrap();
        let (low_res_data, _) = low_res.into_raw_vec_and_offset();
        let low_res: Tensor<Metal, 4> =
            Tensor::from_data(TensorData::new(low_res_data, [1, 256, 48, 48]), &device);

        let fov_encoder = CommonVitModel::new(
            PathBuf::from("/Users/marcintha/workspace/brioche/butter/depthpro_vit_fov.onnx"),
            4,
        )
        .unwrap();

        let mut fov = create_fov_model_with_weight();

        let res = fov.forward(x, low_res, &device, Some(fov_encoder));
        assert!(res.is_ok());

        // @TODO test value, so far quite far i don't really know why
        let res = res.unwrap();
        println!("Final output: {}", res);
        println!("Final output mean: {:?}", res.mean());
    }
}
