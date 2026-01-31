use super::{VitOps, VitResult, utils};
use crate::MixedFloats;
use anyhow::{Result, anyhow};
use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::Transaction;
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor as OrtTensor,
};
use std::path::PathBuf;

#[derive(Debug)]
pub struct PatchVitModel {
    model: Session,
}

impl PatchVitModel {
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

impl VitOps for PatchVitModel {
    fn forward<B: Backend, F: MixedFloats>(
        &mut self,
        input: Tensor<B, 4>,
        device: &B::Device,
    ) -> Result<VitResult<B>> {
        let tensor_data = Transaction::default().register(input).execute();

        let data: Vec<_> = match tensor_data.first() {
            Some(d) => d.to_vec().map_err(|err| {
                anyhow!("Unable to convert the tensor to a vector due to {:?}", err)
            })?,
            None => {
                return Err(anyhow!(
                    "Unable to convert the tensor to a vector due to {:?}",
                    tensor_data
                ));
            }
        };

        let ort_tensor: OrtTensor<F> = OrtTensor::from_array(([35, 3, 384, 384], data))?;
        let output = self
            .model
            .run(ort::inputs!["x" => ort_tensor])
            .map_err(|err| anyhow!("error while running the patch model: {err}"))?;

        let tensor = utils::get_burn_tensor_from_ort::<B, 3, F>(&output, "final_output", device)?;
        let hooks0 = utils::get_burn_tensor_from_ort::<B, 3, F>(&output, "hooks0", device)?;
        let hooks1 = utils::get_burn_tensor_from_ort::<B, 3, F>(&output, "hooks1", device)?;

        Ok(VitResult {
            tensor: tensor,
            hooks0: Some(hooks0),
            hooks1: Some(hooks1),
        })
    }
}
