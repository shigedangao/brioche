use super::{VitOps, VitResult, utils};
use anyhow::{Result, anyhow};
use burn::{Tensor, prelude::Backend};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor as OrtTensor;
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
            .with_optimization_level(GraphOptimizationLevel::Level3)?
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
    fn forward<B: Backend>(
        &mut self,
        input: Tensor<B, 4>,
        device: &B::Device,
    ) -> Result<VitResult<B>> {
        // @TODO use transaction
        let tensor_data: Vec<f32> = input
            .to_data()
            .to_vec()
            .map_err(|err| anyhow!("Unable to convert the tensor to a vector due to {:?}", err))?;

        let tensor: OrtTensor<f32> = OrtTensor::from_array(([1, 3, 384, 384], tensor_data))?;
        let output = self.model.run(ort::inputs!["x" => tensor])?;

        let tensor = utils::get_burn_tensor_from_ort(&output, "tokens", device)?;

        Ok(VitResult {
            tensor,
            hooks0: None,
            hooks1: None,
        })
    }
}
