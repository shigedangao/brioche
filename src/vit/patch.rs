use super::{VitOps, VitResult, utils};
use crate::MixedFloats;
use burn::Tensor;
use burn::prelude::Backend;
use ort::{
    memory::Allocator,
    session::{
        Session,
        builder::{AutoDevicePolicy, GraphOptimizationLevel},
    },
    value::Tensor as OrtTensor,
};
use std::path::PathBuf;

// Constant for the output shape
const OUTPUT_SHAPE: [usize; 3] = [35, 577, 1024];

#[derive(Debug)]
pub struct PatchVitModel {
    model: Session,
}

impl PatchVitModel {
    pub fn new(model_path: PathBuf, thread_nb: usize) -> Result<Self, ort::Error> {
        let model = Session::builder()?
            .with_auto_device(AutoDevicePolicy::MaxPerformance)?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_intra_threads(thread_nb)?
            .commit_from_file(model_path)?;

        Ok(Self { model })
    }
}

impl<B: Backend> VitOps<B> for PatchVitModel {
    fn forward<F: MixedFloats>(
        &mut self,
        input: Tensor<B, 4>,
        device: &B::Device,
    ) -> Result<VitResult<B>, anyhow::Error> {
        let data = input.try_into_data()?.to_vec().map_err(|err| {
            anyhow::anyhow!("Unable to convert the tensor to a vector due to {err}")
        })?;

        let ort_tensor: OrtTensor<F> = OrtTensor::from_array(([35, 3, 384, 384], data))?;

        let mut binding = self.model.create_binding()?;
        binding
            .bind_input("x", &ort_tensor)
            .map_err(|err| anyhow::anyhow!("Unable to bind input due to: {err}"))?;

        // final_output
        binding
            .bind_output(
                "final_output",
                OrtTensor::<F>::new(&Allocator::default(), OUTPUT_SHAPE)?,
            )
            .map_err(|err| anyhow::anyhow!("Unable to bind final_output due to: {err}"))?;

        // hooks0
        binding
            .bind_output(
                "hooks0",
                OrtTensor::<F>::new(&Allocator::default(), OUTPUT_SHAPE)?,
            )
            .map_err(|err| anyhow::anyhow!("Unable to bind hooks0 due to: {err}"))?;

        // hooks1
        binding
            .bind_output(
                "hooks1",
                OrtTensor::<F>::new(&Allocator::default(), OUTPUT_SHAPE)?,
            )
            .map_err(|err| anyhow::anyhow!("Unable to bind hooks1 due to: {err}"))?;

        let output = self
            .model
            .run_binding(&binding)
            .map_err(|err| anyhow::anyhow!("error while running the patch model: {err}"))?;

        let tensor = utils::get_burn_tensor_from_ort::<B, 3, F>(&output, "final_output", device)?;
        let hooks0 = utils::get_burn_tensor_from_ort::<B, 3, F>(&output, "hooks0", device)?;
        let hooks1 = utils::get_burn_tensor_from_ort::<B, 3, F>(&output, "hooks1", device)?;

        Ok(VitResult {
            tensor,
            hooks0: Some(hooks0),
            hooks1: Some(hooks1),
        })
    }
}
