use crate::Brioche;
use crate::MixedFloats;
use crate::brioche_seq::BriocheHeadConfig;
use crate::network::{
    decoder::multires_conv::MultiResDecoderConfig, encoder::EncoderConfig, fov::FovConfig,
};
use crate::utils;
#[cfg(feature = "ort_onnx")]
use crate::vit::{common::CommonVitModel, patch::PatchVitModel};
#[cfg(feature = "burn_onnx")]
use crate::vit::{common_burn::CommonVitModel, patch_burn::PatchVitModel};
use crate::{vit, vit::VitOps};
use anyhow::{Result, anyhow};
use burn::prelude::ToElement;
use burn::tensor::Transaction;
use burn::{prelude::Backend, tensor::Tensor};
use image::{ImageBuffer, Rgb};
use ndarray::Array;
use std::path::PathBuf;

// Constants
const LAST_DIMS: (usize, usize) = (32, 1);
const DIM_DECODER: usize = 256;
const EMBED_DIM: usize = 1024;
const ENCODER_IMG_SIZE: usize = 384 * 4;
const FOV_ENCODER_IMG_SIZE: usize = 384;
const FOV_CONFIG: FovConfig = FovConfig {
    num_features: DIM_DECODER,
    with_fov_encoder: true,
    embed_dim: EMBED_DIM,
};
const BRIOCHE_HEAD_CONFIG: BriocheHeadConfig = BriocheHeadConfig {
    last_dims: LAST_DIMS,
    dim_decoder: DIM_DECODER,
};
const ENCODER_CONFIG: EncoderConfig = EncoderConfig {
    dims_encoder: [256, 512, 1024, 1024],
    patch_encoder_embed_dim: EMBED_DIM,
    image_encoder_embed_dim: EMBED_DIM,
    decoder_features: DIM_DECODER,
    out_size: 384 / 16,
};
const MULTIRES_DECODER_CONFIG: MultiResDecoderConfig = MultiResDecoderConfig {
    dims_encoder: [DIM_DECODER, 256, 512, 1024, 1024],
    dim_decoder: DIM_DECODER,
};

// Custom type to hold the inference output
type InferenceOutput<B> = (ImageBuffer<Rgb<u8>, Vec<u8>>, Option<Tensor<B, 4>>);

/// Runner is a struct which helps to run the depth-pro model
pub struct Four<B: Backend> {
    model: Brioche<B>,
    patch_model: PatchVitModel,
    image_model: CommonVitModel,
    fov_model: CommonVitModel,
    gpu_device: B::Device,
}

/// Configuration in order to run the model
#[derive(Clone, Copy)]
pub struct FourConfig<S: AsRef<str> + Clone> {
    pub patch_vit_path: S,
    pub image_vit_path: S,
    pub fov_vit_path: S,
    pub fov_weight_path: S,
    pub encoder_weight_path: S,
    pub decoder_weight_path: S,
    pub head_weight_path: S,
    pub vit_thread_nb: usize,
}

impl<B: Backend> Four<B> {
    /// Create a new four.
    ///
    /// # Arguments
    /// * `fov_encoder_path` - Path to the fov encoder model
    /// * `patch_vit_path` - Path to the patch vit model
    /// * `image_vit_path` - Path to the image vit model
    /// * `vit_thread_nb` - Number of threads to use for vit models
    /// * `device` - Device to use for the model
    pub fn new<S: AsRef<str> + Clone>(arg: FourConfig<S>) -> Result<Self> {
        let (patch_model, fov_model, image_model) = vit::utils::load_models(arg.clone())?;
        let gpu_device = Default::default();

        // Create the brioche (depth-pro)model
        let bm = Brioche::<B>::new(
            ENCODER_CONFIG,
            MULTIRES_DECODER_CONFIG,
            FOV_CONFIG,
            BRIOCHE_HEAD_CONFIG,
            &gpu_device,
        )?
        .with_record(
            arg.decoder_weight_path,
            arg.encoder_weight_path,
            arg.fov_weight_path,
            arg.head_weight_path,
        )?;

        Ok(Self {
            model: bm,
            patch_model,
            image_model,
            fov_model,
            gpu_device,
        })
    }

    /// Four run the brioche (depth-pro) model and export the output image into a buffer
    pub fn run<S: AsRef<str>, F: MixedFloats>(
        mut self,
        image_path: S,
        is_half_precision: bool,
    ) -> Result<InferenceOutput<B>> {
        let img = image::open(PathBuf::from(image_path.as_ref()))
            .map_err(|err| anyhow!("Unable to load the image due to {err}"))?;
        let input = utils::preprocess_image::<B>(&img, &self.gpu_device, is_half_precision)
            .map_err(|err| anyhow!("Unable to preprocess the image due to {err}"))?;

        // Rescale at the beginning in order to speed up a little bit the inference
        let rescale_input = utils::rescale_image(&img, FOV_ENCODER_IMG_SIZE as u32);
        let fov_input_tensor =
            utils::preprocess_image::<B>(&rescale_input, &self.gpu_device, is_half_precision)
                .map_err(|err| anyhow!("Unable to preprocess the image due to {err}"))?;

        let fov_result = self
            .fov_model
            .forward::<F>(fov_input_tensor.unsqueeze_dim(0), &self.gpu_device)?;

        let (depth, focallength_px) = self.model.infer::<F>(
            (input, fov_result.tensor),
            self.patch_model,
            self.image_model,
            ENCODER_IMG_SIZE,
            &self.gpu_device,
        )?;

        let squeezed_depth: Tensor<B, 2> = depth.detach().squeeze();

        let inverse_depth: Tensor<B, 2> = 1. / squeezed_depth;
        let inverse_depth_min_tensor = inverse_depth.clone().min();
        let inverse_depth_max_tensor = inverse_depth.clone().max();

        let default_min = Tensor::<B, 1>::from_data([1. / 0.1], &self.gpu_device);
        let default_max = Tensor::<B, 1>::from_data([1. / 250.], &self.gpu_device);

        let min_invdepth_vizu_tensor = inverse_depth_min_tensor.min_pair(default_min);
        let max_invdepth_vizu_tensor = inverse_depth_max_tensor.max_pair(default_max);

        let [height, width] = inverse_depth.shape().dims();

        let min_indepth_vizu_scalar = min_invdepth_vizu_tensor.into_scalar().to_f32();
        let max_indepth_vizu_scalar = max_invdepth_vizu_tensor.into_scalar().to_f32();

        let inverse_depth_normalized_tensor = inverse_depth.sub_scalar(min_indepth_vizu_scalar)
            / (max_indepth_vizu_scalar - min_indepth_vizu_scalar);

        let extracted_tensor = Transaction::default()
            .register(inverse_depth_normalized_tensor)
            .execute();

        let Some(inverse_depth_normalized) = extracted_tensor
            .last()
            .and_then(|t| t.to_vec::<f32>().ok())
            .and_then(|t| Array::from_shape_vec((height, width), t).ok())
        else {
            return Err(anyhow!("Unable to get the inverse depth matrix"));
        };

        let inverse_depth_normalized = inverse_depth_normalized.clamp(0., 1.);

        let cmap_matrix = utils::cmap(&inverse_depth_normalized);
        let cmap_matrix = utils::drop_alpha(cmap_matrix);

        let (raw_vec, _) = cmap_matrix.into_raw_vec_and_offset();
        let Some(img_buffer) = ImageBuffer::from_raw(width as u32, height as u32, raw_vec) else {
            return Err(anyhow!("Unable to create image buffer"));
        };

        Ok((img_buffer, focallength_px))
    }
}
