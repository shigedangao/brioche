use anyhow::Result;
use burn::Tensor;
use burn::module::{ModuleMapper, Param};
use burn::prelude::Backend;
use burn::tensor::TensorData;
use image::DynamicImage;
use std::f32::consts::PI;

pub struct DegToRad;

impl<B: Backend> ModuleMapper<B> for DegToRad {
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D>>,
    ) -> Param<burn::Tensor<B, D>> {
        let shape = param.shape();
        let device = param.device();

        let data: Vec<f32> = param
            .to_data()
            .into_vec()
            .expect("Expect to convert the tensor into a vector of float");

        let deg_to_radians = data
            .into_iter()
            .map(|v| v * PI / 180.)
            .collect::<Vec<f32>>();

        let tensor_data = TensorData::new(deg_to_radians, shape);
        let tensor = Tensor::from_floats(tensor_data, &device);

        Param::from_tensor(tensor)
    }
}

/// Preprocess an image by converting it to RGB, normalizing pixel values, and reshaping it.
/// This step reprseents the following set of "functions"
///
///  transform = Compose(
///       [
///           ToTensor(),
///           Lambda(lambda x: x.to(device)), <-- not needed in Burn
///           Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
///           ConvertImageDtype(precision), <-- not needed in Burn
///       ]
///   )
///
/// # Arguments
/// * `img` - The image to preprocess.
/// * `device` - The device to place the tensor on.
pub fn preprocess_image<B: Backend>(
    img: &DynamicImage,
    device: &B::Device,
) -> Result<Tensor<B, 3>> {
    // Convert the image to rgb if needed
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    // Convert to float vector (equivalent ToTensor)
    let pixels: Vec<f32> = rgb_img
        .pixels()
        .flat_map(|p| {
            // Normalize from [0, 255] to [0.0, 1.0]
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    let tensor_data = TensorData::new(pixels, [3, height as usize, width as usize]);
    let tensor = Tensor::from_floats(tensor_data, device);

    // normalize the tensor
    let tensor = (tensor - 0.5) / 0.5;

    Ok(tensor)
}
