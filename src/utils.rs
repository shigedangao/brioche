use burn::prelude::Backend;
use burn::tensor::TensorData;
use burn::{Tensor, tensor::FloatDType};
use colorgrad::Gradient;
use image::DynamicImage;
use ndarray::Zip;
use ndarray::{Array2, Array3, s};

/// Preprocess an image by converting it to RGB, normalizing pixel values, and reshaping it.
/// This step reprseents the following set of "functions"
///
///  transform = Compose(
///       [
///           `ToTensor()`,
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
    is_half_precision: bool,
) -> Tensor<B, 3> {
    let rgb_img = img.to_rgb32f();
    let (width, height) = rgb_img.dimensions();

    let raw = rgb_img.into_raw();
    let tensor_data = TensorData::new(raw, [height as usize, width as usize, 3]);
    let tensor = Tensor::from_floats(tensor_data, device).permute([2, 0, 1]);

    // Create mean and std as 1D tensors & reshape to (3, 1, 1) for broadcasting across H and W dimensions
    let mean = Tensor::<B, 1>::from_floats([0.5, 0.5, 0.5], device).reshape([3, 1, 1]);
    let std = Tensor::<B, 1>::from_floats([0.5, 0.5, 0.5], device).reshape([3, 1, 1]);

    // normalize the tensor
    let mut tensor = (tensor - mean) / std;

    if is_half_precision {
        tensor = tensor.cast(FloatDType::F16);
    }

    tensor
}

/// Rescale image to encoder base size
///
/// # Arguments
/// * `img` - The image to rescale
/// * `encoder_base_size` - The base size to rescale the image to
pub fn rescale_image(img: &DynamicImage, encoder_base_size: u32) -> DynamicImage {
    img.resize_exact(
        encoder_base_size,
        encoder_base_size,
        image::imageops::FilterType::Lanczos3,
    )
}

/// Convert depth map to color map. Perform the cmap operation that is being used in the matplotlib library.
///
/// # Arguments
/// * `input` - The depth map to convert
pub fn cmap(mut input: Array2<f32>) -> Array3<u8> {
    let (h, w) = input.dim();
    // Create the turbo gradient domain [0..1]
    let grad = colorgrad::preset::turbo();

    // Create a new matrix with the proper shape
    let mut rgb = Array3::<u8>::zeros((h, w, 4));
    Zip::indexed(&mut input).for_each(|(y, x), v| {
        let rgba = grad.at(v.clamp(0., 1.)).to_rgba8();

        rgb[[y, x, 0]] = rgba[0];
        rgb[[y, x, 1]] = rgba[1];
        rgb[[y, x, 2]] = rgba[2];
        rgb[[y, x, 3]] = rgba[3];
    });

    rgb
}

/// Drop the alpha channel from an RGBA image
///
/// # Arguments
/// * `rgba` - The RGBA image to drop the alpha channel from
pub fn drop_alpha(rgba: Array3<u8>) -> Array3<u8> {
    rgba.slice(s![.., .., 0..3]).to_owned() // (H, W, 3)
}
