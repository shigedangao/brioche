use crate::model::image::Model as ImageModel;
use crate::vit::VitOps;
use crate::{model::fov::Model as FovModel, vit::VitResult};
use anyhow::Result;
use burn::prelude::Backend;

/// CommonVitModelListRepresents the kind of common vit model to use.
pub enum CommonVitModelList {
    Fov,
    Image,
}

/// CommonVitModel is a wrapper around the Fov and Image models that can be used interchangeably.
pub struct CommonVitModel {
    kind: CommonVitModelList,
}

impl CommonVitModel {
    /// Creates a new CommonVitModel with the given kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - The kind of model to use.
    pub fn new(kind: CommonVitModelList) -> Result<Self> {
        Ok(Self { kind })
    }
}

impl<B: Backend> VitOps<B> for CommonVitModel {
    fn forward<F: crate::MixedFloats>(
        &mut self,
        input: burn::Tensor<B, 4>,
        _: &B::Device,
    ) -> Result<super::VitResult<B>> {
        let tensor = match self.kind {
            CommonVitModelList::Fov => FovModel::<B>::default().forward(input),
            CommonVitModelList::Image => ImageModel::<B>::default().forward(input),
        };

        Ok(VitResult {
            tensor,
            hooks0: None,
            hooks1: None,
        })
    }
}
