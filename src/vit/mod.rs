use crate::MixedFloats;
use crate::four::FourConfig;
use anyhow::Result;
use burn::{Tensor, prelude::Backend};
#[cfg(feature = "ort_onnx")]
use ort::value::Shape;
use std::thread;

#[cfg(feature = "ort_onnx")]
pub mod common;
#[cfg(feature = "ort_onnx")]
pub mod patch;

#[cfg(feature = "burn_onnx")]
pub mod common_burn;
#[cfg(feature = "burn_onnx")]
pub mod patch_burn;

#[derive(Debug)]
pub struct VitResult<B: Backend> {
    pub tensor: Tensor<B, 3>,
    pub hooks0: Option<Tensor<B, 3>>,
    pub hooks1: Option<Tensor<B, 3>>,
}

pub trait VitOps<B: Backend> {
    /// Forward pass of the ViT model.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch_size, channels, height, width).
    /// * `device` - Device on which the computation should be performed.
    ///
    /// # Returns
    /// A `Result` containing the output tensor of shape (batch_size, num_patches, embedding_dim),
    /// and two optional hooks for intermediate results.
    fn forward<F: MixedFloats>(
        &mut self,
        input: Tensor<B, 4>,
        device: &B::Device,
    ) -> Result<VitResult<B>>;
}

pub(crate) mod utils {
    use super::*;
    #[cfg(feature = "burn_onnx")]
    use crate::vit::{common_burn::CommonVitModel, patch_burn::PatchVitModel};
    #[cfg(feature = "ort_onnx")]
    use {
        crate::{
            MixedFloats,
            vit::{common::CommonVitModel, patch::PatchVitModel},
        },
        burn::tensor::{Shape as BurnShape, TensorData},
        ort::session::SessionOutputs,
    };

    /// Extracts the shape of a tensor into a fixed-size array.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor.
    ///
    /// # Returns
    /// A `Result` containing the shape as a fixed-size array.
    #[cfg(feature = "ort_onnx")]
    pub fn extract_tensor_shape<const S: usize>(shape: &Shape) -> Result<[usize; S]> {
        if shape.len() != S {
            return Err(anyhow::anyhow!("Unexpected shape for tokens: {:?}", shape));
        }

        let v: Vec<usize> = shape.iter().map(|v| *v as usize).collect();

        v.try_into()
            .map_err(|err| anyhow::anyhow!("Unable to convert shape into desired slice: {:?}", err))
    }

    /// Get a burn tensor from ONNX Runtime output.
    ///
    /// # Arguments
    /// * `output` - SessionOutputs<'a>.
    /// * `output_ident` - The identifier of the output.
    /// * `device` - The device to allocate the tensor on.
    ///
    /// # Returns
    /// A `Result` containing the burn tensor.
    #[cfg(feature = "ort_onnx")]
    pub fn get_burn_tensor_from_ort<B: Backend, const S: usize, F: MixedFloats>(
        output: &SessionOutputs,
        output_ident: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, S>> {
        let tensor = match output.get(output_ident) {
            Some(output) => {
                let (shape, data) = output.try_extract_tensor::<F>()?;
                let shape_slice = utils::extract_tensor_shape::<S>(shape)?;
                let tensor_data = TensorData::new(data.to_vec(), BurnShape::new(shape_slice));

                Tensor::<B, S>::from_floats(tensor_data, device)
            }
            None => {
                return Err(anyhow::anyhow!(
                    "Failed to extract {output_ident} from model output"
                ));
            }
        };

        Ok(tensor)
    }

    #[cfg(feature = "burn_onnx")]
    pub fn load_models<S: AsRef<str> + Clone>(
        _: FourConfig<S>,
    ) -> Result<(PatchVitModel, CommonVitModel, CommonVitModel)> {
        use crate::vit::common_burn::CommonVitModelList;

        Ok((
            PatchVitModel::new()?,
            CommonVitModel::new(common_burn::CommonVitModelList::Fov)?,
            CommonVitModel::new(CommonVitModelList::Image)?,
        ))
    }

    #[cfg(feature = "ort_onnx")]
    pub fn load_models<S: AsRef<str> + Clone + Send + Sync>(
        config: FourConfig<S>,
    ) -> Result<(PatchVitModel, CommonVitModel, CommonVitModel)> {
        let handles = thread::scope(|scope| {
            let patch_model = scope.spawn(|| {
                PatchVitModel::new(
                    std::path::PathBuf::from(config.patch_vit_path.as_ref()),
                    config.vit_thread_nb,
                )
            });

            let fov_model = scope.spawn(|| {
                CommonVitModel::new(
                    std::path::PathBuf::from(config.fov_vit_path.as_ref()),
                    config.vit_thread_nb,
                )
            });

            let image_model = scope.spawn(|| {
                CommonVitModel::new(
                    std::path::PathBuf::from(config.image_vit_path.as_ref()),
                    config.vit_thread_nb,
                )
            });

            (patch_model.join(), fov_model.join(), image_model.join())
        });

        let (p_handle, f_handle, i_handle) = handles;

        let patch_model =
            p_handle.map_err(|_| anyhow::anyhow!("Unable to get the handle for patch model"))??;

        let fov_model =
            f_handle.map_err(|_| anyhow::anyhow!("Unable to get the handle for fov model"))??;

        let image_model =
            i_handle.map_err(|_| anyhow::anyhow!("Unable to get the handle for image model"))??;

        Ok((patch_model, fov_model, image_model))
    }
}
