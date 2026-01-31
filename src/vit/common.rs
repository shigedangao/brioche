use super::{VitOps, VitResult, utils};
use crate::MixedFloats;
use anyhow::{Result, anyhow};
use burn::{Tensor, prelude::Backend};
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor as OrtTensor,
};
use std::path::PathBuf;

/// CommonVitModel represents a Vision Transformer (ViT) model for feature extraction that is being used by Depth-pro
///
/// /!\ This basically represent the FovEncoder & ImageEncoder module that is being used in the fov.rs (encoder argument).
///     We use ort to load the model + weight and perform the forward pass using the provided backend.
#[derive(Debug)]
pub struct CommonVitModel {
    model: Session,
}

impl CommonVitModel {
    /// Create a new CommonVitModel instance.
    /// /!\ The fov model needs to be passed. ORT will load the model + weight.
    ///     As a result the model + weight needs to be directory.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model file.
    /// * `thread_nb` - Number of threads to use for inference.
    pub fn new(model_path: PathBuf, thread_nb: usize) -> Result<Self> {
        let model = Session::builder()?
            .with_execution_providers([
                // Prefer coreml for apple devices
                ep::CoreML::default().build(),
                // Prefer cuda for gpu devices
                ep::CUDA::default().build(),
                // Prefer directml for windows devices
                ep::DirectML::default().build(),
            ])?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_intra_threads(thread_nb)?
            .commit_from_file(model_path)?;

        Ok(Self { model })
    }
}

impl VitOps for CommonVitModel {
    /// Perform a forward pass on the input data.
    ///
    ///
    /// # Arguments
    /// * `input` - Input data.
    /// * `device` - Device to use for inference.
    fn forward<B: Backend, F: MixedFloats>(
        &mut self,
        input: Tensor<B, 4>,
        device: &B::Device,
    ) -> Result<VitResult<B>> {
        // /!\ Some overhead happened when performing this operation for the FOV tensor.
        let data: Vec<F> = input
            .to_data()
            .to_vec()
            .map_err(|err| anyhow!("Unable to convert the tensor to a vector due to {:?}", err))?;

        let tensor: OrtTensor<F> = OrtTensor::from_array(([1, 3, 384, 384], data))?;
        let output = self.model.run(ort::inputs!["x" => tensor])?;

        let tensor = utils::get_burn_tensor_from_ort::<B, 3, F>(&output, "tokens", device)?;

        Ok(VitResult {
            tensor,
            hooks0: None,
            hooks1: None,
        })
    }
}
