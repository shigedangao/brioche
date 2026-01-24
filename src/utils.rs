use anyhow::Result;
use burn::Tensor;
use burn::module::{ModuleMapper, Param};
use burn::prelude::Backend;
use burn::tensor::TensorData;
use colorgrad::Gradient;
use image::DynamicImage;
use ndarray::{Array2, Array3, s};
use std::f32::consts::PI;

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
    let rgb_img = img.to_rgb32f();
    let (width, height) = rgb_img.dimensions();

    // Create a vector to store each channel
    let mut r_channel = Vec::with_capacity((width * height) as usize);
    let mut g_channel = Vec::with_capacity((width * height) as usize);
    let mut b_channel = Vec::with_capacity((width * height) as usize);

    for pixel in rgb_img.pixels() {
        r_channel.push(pixel[0]);
        g_channel.push(pixel[1]);
        b_channel.push(pixel[2]);
    }

    // Concatenate channels in CHW order
    let mut pixels = Vec::with_capacity((3 * width * height) as usize);
    pixels.extend(r_channel);
    pixels.extend(g_channel);
    pixels.extend(b_channel);

    let tensor_data = TensorData::new(pixels, [3, height as usize, width as usize]);
    let tensor = Tensor::from_floats(tensor_data, device);

    // Create mean and std as 1D tensors
    let mean = Tensor::<B, 1>::from_floats([0.5, 0.5, 0.5], &device);
    let std = Tensor::<B, 1>::from_floats([0.5, 0.5, 0.5], &device);

    // Reshape to (3, 1, 1) for broadcasting across H and W dimensions
    let mean = mean.reshape([3, 1, 1]);
    let std = std.reshape([3, 1, 1]);

    // normalize the tensor
    let tensor = (tensor - mean) / std;

    Ok(tensor)
}

pub fn cmap(input: &Array2<f32>) -> Array3<u8> {
    let (h, w) = input.dim();
    // Create the turbo gradient domain [0..1]
    let grad = colorgrad::preset::turbo();

    // Create a new matrix with the proper shape
    let mut rgb = Array3::<u8>::zeros((h, w, 4));

    // Copy depth into each channel
    for ((y, x), v) in input.indexed_iter() {
        let v = v.clamp(0.0, 1.0); // important: match cmap domain
        let rgba = grad.at(v).to_rgba8(); // [r,g,b,a] u8 :contentReference[oaicite:2]{index=2}

        rgb[[y, x, 0]] = rgba[0];
        rgb[[y, x, 1]] = rgba[1];
        rgb[[y, x, 2]] = rgba[2];
        rgb[[y, x, 3]] = rgba[3];
    }

    rgb
}

pub fn drop_alpha(rgba: Array3<u8>) -> Array3<u8> {
    rgba.slice(s![.., .., 0..3]).to_owned() // (H, W, 3)
}
