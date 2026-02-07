use crate::MixedFloats;
use anyhow::{Result, anyhow};
use burn::{
    Tensor,
    prelude::Backend,
    tensor::{Shape as BurnShape, TensorData},
};
use ort::tensor::Shape;

pub mod common;
pub mod patch;

#[derive(Debug)]
pub struct VitResult<B: Backend> {
    pub tensor: Tensor<B, 3>,
    pub hooks0: Option<Tensor<B, 3>>,
    pub hooks1: Option<Tensor<B, 3>>,
}

pub trait VitOps {
    /// Forward pass of the ViT model.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch_size, channels, height, width).
    /// * `device` - Device on which the computation should be performed.
    ///
    /// # Returns
    /// A `Result` containing the output tensor of shape (batch_size, num_patches, embedding_dim),
    /// and two optional hooks for intermediate results.
    fn forward<B: Backend, F: MixedFloats>(
        &mut self,
        input: Tensor<B, 4>,
        device: &B::Device,
    ) -> Result<VitResult<B>>;
}

mod utils {
    use crate::MixedFloats;

    use super::*;
    use ort::session::SessionOutputs;

    /// Extracts the shape of a tensor into a fixed-size array.
    ///
    /// # Arguments
    /// * `shape` - The shape of the tensor.
    ///
    /// # Returns
    /// A `Result` containing the shape as a fixed-size array.
    pub fn extract_tensor_shape<const S: usize>(shape: &Shape) -> Result<[usize; S]> {
        if shape.len() != S {
            return Err(anyhow!("Unexpected shape for tokens: {:?}", shape));
        }

        let v: Vec<usize> = shape.iter().map(|v| *v as usize).collect();

        v.try_into()
            .map_err(|err| anyhow!("Unable to convert shape into desired slice: {:?}", err))
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
                return Err(anyhow!(
                    "Failed to extract {output_ident} from model output"
                ));
            }
        };

        Ok(tensor)
    }
}
