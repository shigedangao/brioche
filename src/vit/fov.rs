use anyhow::{Result, anyhow};
use burn::tensor::{Shape, TensorData};
use burn::{Tensor, prelude::Backend};
use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor as OrtTensor;
use std::path::PathBuf;

/// FovVitModel represents a Vision Transformer (ViT) model for feature extraction that is being used by Depth-pro
///
/// /!\ This basically represent the FovEncoder module that is being used in the fov.rs (encoder argument).
///     We use ort to load the model + weight and perform the forward pass using the provided backend.
#[derive(Debug)]
pub struct FovVitModel {
    model: Session,
}

impl FovVitModel {
    /// Create a new FovVitModel instance.
    /// /!\ The fov model needs to be passed. ORT will load the model + weight.
    ///     As a result the model + weight needs to be directory.
    ///
    /// # Arguments
    /// * `model_path` - Path to the model file.
    /// * `thread_nb` - Number of threads to use for inference.
    pub fn new<B: Backend>(model_path: PathBuf, thread_nb: usize) -> Result<Self> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(thread_nb)?
            .commit_from_file(model_path)?;

        Ok(Self { model })
    }

    /// Perform a forward pass on the input data.
    ///
    /// /!\ Note that the input data is of type Array4<f32> which uses ndarray.
    ///
    /// # Arguments
    /// * `input` - Input data.
    /// * `device` - Device to use for inference.
    pub fn forward<B: Backend>(
        &mut self,
        input: Array4<f32>,
        device: &B::Device,
    ) -> Result<Tensor<B, 3>> {
        let tensor: OrtTensor<f32> = OrtTensor::from_array(input)?;
        let output = self.model.run(ort::inputs!["x" => tensor])?;

        let tensor = match output.get("tokens") {
            Some(output) => {
                let (shape, data) = output.try_extract_tensor::<f32>()?;
                let tensor_data = TensorData::new(
                    data.to_vec(),
                    // @TODO use a safe way to get the size of the shape
                    Shape::new([shape[0] as usize, shape[1] as usize, shape[2] as usize]),
                );

                let tensor = Tensor::<B, 3>::from_floats(tensor_data, &device);

                tensor
            }
            None => {
                return Err(anyhow!("Failed to extract tokens from model output"));
            }
        };

        Ok(tensor)
    }
}
