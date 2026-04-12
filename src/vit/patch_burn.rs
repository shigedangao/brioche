use crate::{
    model::patch::Model,
    vit::{VitOps, VitResult},
};
use anyhow::Result;
use burn::prelude::Backend;

/// PatchVitModel is a wrapper around the patch model that implements the VitOps trait.
pub struct PatchVitModel {}

impl PatchVitModel {
    /// Create a new PatchVitModel instance.
    pub fn new() -> Result<Self> {
        Ok(PatchVitModel {})
    }
}

impl<B: Backend> VitOps<B> for PatchVitModel {
    fn forward<F: crate::MixedFloats>(
        &mut self,
        input: burn::Tensor<B, 4>,
        _: &<B as Backend>::Device,
    ) -> Result<super::VitResult<B>> {
        let model = Model::<B>::default();

        let (tensor, h0, h1) = model.forward(input);

        Ok(VitResult {
            tensor,
            hooks0: Some(h0),
            hooks1: Some(h1),
        })
    }
}
