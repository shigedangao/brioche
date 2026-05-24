use super::{VitOps, VitResult, utils};
use crate::MixedFloats;
use anyhow::anyhow;
use burn::{prelude::Backend, tensor::Tensor};
use ort::{
    memory::Allocator,
    session::{
        Session,
        builder::{AutoDevicePolicy, GraphOptimizationLevel},
    },
    value::Tensor as OrtTensor,
};
use std::path::PathBuf;

/// `CommonVitModel` represents a Vision Transformer (`ViT`) model for feature extraction that is being used by Depth-pro
///
/// /!\ This basically represent the `FovEncoder` & `ImageEncoder` module that is being used in the fov.rs (encoder argument).
///     We use ort to load the model + weight and perform the forward pass using the provided backend.
#[derive(Debug)]
pub struct CommonVitModel {
    model: Session,
}

impl CommonVitModel {
    /// Create a new `CommonVitModel` instance.
    /// /!\ The fov model needs to be passed. ORT will load the model + weight.
    ///     As a result the model + weight needs to be directory.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model file.
    /// * `thread_nb` - Number of threads to use for inference.
    pub fn new(model_path: PathBuf, thread_nb: usize) -> Result<Self, ort::Error> {
        let model = Session::builder()?
            .with_auto_device(AutoDevicePolicy::MaxPerformance)?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_intra_threads(thread_nb)?
            .commit_from_file(model_path)?;

        Ok(Self { model })
    }
}

impl<B: Backend> VitOps<B> for CommonVitModel {
    /// Perform a forward pass on the input data.
    ///
    ///
    /// # Arguments
    /// * `input` - Input data.
    /// * `device` - Device to use for inference.
    fn forward<F: MixedFloats>(
        &mut self,
        input: Tensor<B, 4>,
        device: &B::Device,
    ) -> Result<VitResult<B>, anyhow::Error> {
        // /!\ Some overhead happened when performing this operation for the FOV tensor.
        let data = input
            .try_into_data()?
            .to_vec()
            .map_err(|err| anyhow!("Unable to convert the tensor to a vector due to {err}"))?;

        let tensor: OrtTensor<F> = OrtTensor::from_array(([1, 3, 384, 384], data))?;

        let mut binding = self.model.create_binding()?;
        binding
            .bind_input("x", &tensor)
            .map_err(|err| anyhow!("Unable to bind input due to: {err}"))?;

        binding
            .bind_output(
                "tokens",
                OrtTensor::<F>::new(&Allocator::default(), [1_usize, 577_usize, 1024_usize])?,
            )
            .map_err(|err| anyhow!("Unable to bind output due to: {err}"))?;

        let output = self.model.run_binding(&binding)?;
        let tensor = utils::get_burn_tensor_from_ort::<B, 3, F>(&output, "tokens", device)?;

        Ok(VitResult {
            tensor,
            hooks0: None,
            hooks1: None,
        })
    }
}
