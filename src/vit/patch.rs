use std::path::PathBuf;

use super::{VitOps, VitResult, utils};
use anyhow::{Result, anyhow};
use burn::Tensor;
use burn::prelude::Backend;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor as OrtTensor;

#[derive(Debug)]
pub struct PatchVitModel {
    model: Session,
}

impl PatchVitModel {
    pub fn new(model_path: PathBuf, thread_nb: usize) -> Result<Self> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(thread_nb)?
            .commit_from_file(model_path)?;

        Ok(Self { model })
    }
}

impl VitOps for PatchVitModel {
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

        // @TODO replace with shape of actual tensor
        let ort_tensor: OrtTensor<f32> = OrtTensor::from_array(([35, 3, 384, 384], tensor_data))?;
        let output = self.model.run(ort::inputs!["x" => ort_tensor])?;

        let tensor = utils::get_burn_tensor_from_ort(&output, "final_output", device)?;
        let hooks0 = utils::get_burn_tensor_from_ort(&output, "hooks0", device)?;
        let hooks1 = utils::get_burn_tensor_from_ort(&output, "hooks1", device)?;

        Ok(VitResult {
            tensor: tensor,
            hooks0: Some(hooks0),
            hooks1: Some(hooks1),
        })
    }
}
