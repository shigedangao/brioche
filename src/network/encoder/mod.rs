use crate::vit::{VitOps, common::CommonVitModel, patch::PatchVitModel};
use anyhow::{Result, anyhow};
use burn::{
    Tensor,
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        interpolate::{Interpolate2dConfig, InterpolateMode},
    },
    prelude::Backend,
};
use projection_seq::ProjectionSeq;

mod projection_seq;

// Constants
const PATCH_SIZE: usize = 384;

// Compute patch stride at compile time
//
// # Arguments
// * `overlap_ratio` - The overlap ratio between patches.
const fn compute_patch_stride(overlap_ratio: f64) -> usize {
    (PATCH_SIZE as f64 * (1. - overlap_ratio)) as usize
}

/// Encoder represent the depth-pro encoder. The implementation refer to the following original python file
/// @link https://github.com/apple/ml-depth-pro/blob/9efe5c1def37a26c5367a71df664b18e1306c708/src/depth_pro/network/encoder.py#L14
///
/// /!\ In the depth-pro implementation. Most of the upsample layers are represented by an array of a mix of Conv2d and ConvTranspose2d layers. As in Rust this is not possible to perform that. We wrap this operation in a ProjectionSeq struct.
#[derive(Debug, Module)]
pub struct Encoder<B: Backend> {
    upsample_latent0: ProjectionSeq<B>,
    upsample_latent1: ProjectionSeq<B>,
    upsample0: ProjectionSeq<B>,
    upsample1: ProjectionSeq<B>,
    upsample2: ProjectionSeq<B>,
    upsample_lowres: ConvTranspose2d<B>,
    fuse_lowres: Conv2d<B>,
    out_size: usize,
}

#[derive(Debug)]
pub struct EncoderOutput<B: Backend> {
    pub x_latent0_features: Tensor<B, 4>,
    pub x_latent1_features: Tensor<B, 4>,
    pub x0_features: Tensor<B, 4>,
    pub x1_features: Tensor<B, 4>,
    pub x_global_features: Tensor<B, 4>,
}

#[derive(Debug, Default)]
pub struct EncoderConfig {
    dims_encoder: Vec<usize>,
    patch_encoder_embed_dim: usize,
    image_encoder_embed_dim: usize,
    decoder_features: usize,
    out_size: usize,
}

impl<B: Backend> Encoder<B> {
    /// Create a new encoder.
    ///
    /// # Arguments
    /// * `dims_encoder` - The dimensions of the encoder.
    /// * `patch_encoder_embed_dim` - The patch encoder embedding dimension.
    /// * `image_encoder_embed_dim` - The image encoder embedding dimension.
    /// * `decoder_features` - The decoder features.
    /// * `device` - The device to use.
    /// * `out_size` - The output size.
    pub fn new(config: EncoderConfig, device: &B::Device) -> Self {
        let EncoderConfig {
            dims_encoder,
            patch_encoder_embed_dim,
            image_encoder_embed_dim,
            decoder_features,
            out_size,
        } = config;

        let upsample_latent0 = ProjectionSeq::new(
            patch_encoder_embed_dim,
            dims_encoder.first().copied(),
            decoder_features,
            3,
            device,
        );

        let upsample_latent1 = ProjectionSeq::new(
            patch_encoder_embed_dim,
            None,
            dims_encoder.first().copied().unwrap_or_default(),
            2,
            device,
        );

        let upsample0 = ProjectionSeq::new(
            patch_encoder_embed_dim,
            None,
            dims_encoder.get(1).copied().unwrap_or_default(),
            1,
            device,
        );

        let upsample1 = ProjectionSeq::new(
            patch_encoder_embed_dim,
            None,
            dims_encoder.get(2).copied().unwrap_or_default(),
            1,
            device,
        );

        let upsample2 = ProjectionSeq::new(
            patch_encoder_embed_dim,
            None,
            dims_encoder.get(3).copied().unwrap_or_default(),
            1,
            device,
        );

        let dims_encoder_three = dims_encoder.get(3).copied().unwrap_or_default();

        let upsample_lowres =
            ConvTranspose2dConfig::new([image_encoder_embed_dim, dims_encoder_three], [2, 2])
                .with_stride([2, 2])
                .with_padding([0, 0])
                .with_bias(true)
                .init::<B>(&device);

        let fuse_lowres = Conv2dConfig::new(
            [dims_encoder_three + dims_encoder_three, dims_encoder_three],
            [1, 1],
        )
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(0, 0))
        .init::<B>(device);

        Self {
            upsample_latent0,
            upsample_latent1,
            upsample0,
            upsample1,
            upsample2,
            upsample_lowres,
            fuse_lowres,
            out_size,
        }
    }

    /// Create a pyramid of tensors.
    ///
    /// # Arguments
    /// * `x` - The input tensor.
    fn create_pyramid(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let x0 = x.clone();

        let x1_interpolate = Interpolate2dConfig::new()
            .with_scale_factor(Some([0.5, 0.5]))
            .with_mode(InterpolateMode::Linear)
            .with_output_size(None)
            .init();

        let x2_interpolate = Interpolate2dConfig::new()
            .with_scale_factor(Some([0.25, 0.25]))
            .with_mode(InterpolateMode::Linear)
            .with_output_size(None)
            .init();

        (
            // Original resolution: 1536 by default.
            x,
            // Half resolution: 768 by default.
            x1_interpolate.forward(x0.clone()),
            // Quarter resolution: 384 by default. corresponding to the backbone resolution.
            x2_interpolate.forward(x0),
        )
    }

    /// Split the input tensor into patches.
    ///
    /// # Arguments
    /// * `input` - The input tensor.
    /// * `patch_stride` - The patch stride.
    fn split(&self, input: Tensor<B, 4>, patch_stride: usize) -> Result<Tensor<B, 4>> {
        let Some(ref image_size) = input.shape().last().copied() else {
            return Err(anyhow!(
                "Unable to get the image size from the shape of the tensor"
            ));
        };

        let [batch, chan, _, _] = input.dims();
        let steps = (image_size - PATCH_SIZE + patch_stride - 1) / patch_stride + 1;

        let mut patches = Vec::with_capacity(steps * steps);
        // process height
        for j in 0..steps {
            let j0 = j * patch_stride;
            let j1 = j0 + PATCH_SIZE;

            // process width
            for i in 0..steps {
                let i0 = i * patch_stride;
                let i1 = i0 + PATCH_SIZE;

                let patch = input.clone().slice([0..batch, 0..chan, j0..j1, i0..i1]);
                patches.push(patch);
            }
        }

        Ok(Tensor::cat(patches, 0))
    }

    /// Reshape the input tensor into a feature map.
    ///
    /// # Arguments
    /// * `input` - The input tensor.
    /// * `width` - The width of the feature map.
    /// * `height` - The height of the feature map.
    fn reshape_feature(
        &self,
        embeddings: Tensor<B, 3>,
        width: usize,
        height: usize,
    ) -> Tensor<B, 4> {
        let [batch, hw, ch] = embeddings.shape().dims();

        let embeddings_slice = embeddings.slice([0..batch, 1..hw, 0..ch]);
        let reshaped_embeddings = embeddings_slice
            .reshape([batch, height, width, ch])
            .permute([0, 3, 1, 2]);

        reshaped_embeddings
    }

    /// Merge the input tensor into a feature map.
    ///
    /// # Arguments
    /// * `input` - The input tensor.
    /// * `batch_size` - The batch size.
    /// * `padding` - The padding.
    fn merge(
        &self,
        input: Tensor<B, 4>,
        batch_size: usize,
        padding: usize,
    ) -> Result<Tensor<B, 4>> {
        let [b, chan, height, width] = input.shape().dims();

        let steps = (b / batch_size).isqrt();
        let mut idx = 0;

        let mut output_list = Vec::with_capacity(steps);
        for j in 0..steps {
            let mut output_row_list = Vec::with_capacity(steps);

            for i in 0..steps {
                // The cloning is necessary as we need to slice the tensor multiple times
                // /!\ Here we could not use Rc or Arc as the tensor is not shared and we need to clone it (consume self)
                let mut output = input.clone().slice([
                    batch_size * idx..batch_size * (idx + 1),
                    0..chan,
                    0..height,
                    0..width,
                ]);

                if j != 0 {
                    // /!\ Get the shape of the output tensor before each slicing as the shape change after each slicing operation
                    //     pytorch may be doing this under the hood.
                    let [ob, oc, oh, ow] = output.shape().dims();
                    output = output.slice([0..ob, 0..oc, padding..oh, 0..ow]);
                }
                if i != 0 {
                    let [ob, oc, oh, ow] = output.shape().dims();
                    output = output.slice([0..ob, 0..oc, 0..oh, padding..ow]);
                }
                if j != steps - 1 {
                    let [ob, oc, oh, ow] = output.shape().dims();
                    output = output.slice([0..ob, 0..oc, 0..oh - padding, 0..ow]);
                }
                if i != steps - 1 {
                    let [ob, oc, oh, ow] = output.shape().dims();
                    output = output.slice([0..ob, 0..oc, 0..oh, 0..ow - padding]);
                }

                output_row_list.push(output);
                idx += 1;
            }

            // Concatenate the rows to form the output tensor on the width dimension.
            let output_row = Tensor::cat(output_row_list, 3);
            output_list.push(output_row);
        }

        Ok(Tensor::cat(output_list, 2))
    }

    /// Compute the forward pass of the encoder.
    ///
    /// # Arguments
    /// * `input` - The input tensor.
    /// * `batch_size` - The batch size.
    /// * `padding` - The padding.
    /// * `device` - The device.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        mut patch_encoder: PatchVitModel,
        mut image_encoder: CommonVitModel,
        device: &B::Device,
    ) -> Result<EncoderOutput<B>> {
        let Some(batch_size) = input.shape().first().copied() else {
            return Err(anyhow!("Unable to determine batch size"));
        };

        // Step 0: create a 3-level image pyramid.
        // x2_patches -> 1x1 # 384x384 at the lowest resolution (384x384).
        let (x0, x1, x2_patches) = self.create_pyramid(input.clone());

        // Step 1: split to create batched overlapped mini-images at the backbone (BeiT/ViT/Dino)
        // resolution.
        // 5x5 @ 384x384 at the highest resolution (1536x1536).
        let x0_patches = self.split(x0, compute_patch_stride(0.25))?;
        // 3x3 @ 384x384 at the middle resolution (768x768).
        let x1_patches = self.split(x1, compute_patch_stride(0.5))?;

        // These 3 variables are the batch sizes of the patches. Use them to split the patches into chunks.
        let x0_b = x0_patches.shape().first().copied().unwrap_or_default();
        let x1_b = x1_patches.shape().first().copied().unwrap_or_default();
        let x2_b = x2_patches.shape().first().copied().unwrap_or_default();

        // Concatenate all the sliding window patches and form a batch of size (35=5x5+3x3+1x1).
        let x_pyramid_patches = Tensor::cat(vec![x0_patches, x1_patches, x2_patches.clone()], 0);

        // Step 2: Run the backbone (BeiT) model and get the result of large batch size.
        let (x_pyramid_encoding, bb_highres_hook0, bb_highres_hook1) = patch_encoder
            .forward(x_pyramid_patches, device)
            .and_then(|x| {
                Ok((
                    self.reshape_feature(x.tensor, self.out_size, self.out_size),
                    x.hooks0,
                    x.hooks1,
                ))
            })?;

        if bb_highres_hook0.is_none() || bb_highres_hook1.is_none() {
            return Err(anyhow!("Highres hooks are missing"));
        }

        // Step 3: Merging
        // 3.1 Merge highres latent encoding.
        let target_batch_size = batch_size * 5 * 5;

        let x_latent0_encodings =
            self.reshape_feature(bb_highres_hook0.unwrap(), self.out_size, self.out_size);
        let [x0, c0, h0, w0] = x_latent0_encodings.shape().dims();

        // Using target_batch_size.min(w0) for width in order not to exceed the batch size
        let x0_latent_features = self.merge(
            x_latent0_encodings.slice([0..x0, 0..c0, 0..h0, 0..target_batch_size.min(w0)]),
            batch_size,
            3,
        )?;

        // Using target_batch_size.min(w1) for width in order not to exceed the batch size
        let x_latent1_encodings =
            self.reshape_feature(bb_highres_hook1.unwrap(), self.out_size, self.out_size);
        let [x1, c1, h1, w1] = x_latent1_encodings.shape().dims();
        let x1_latent_features = self.merge(
            x_latent1_encodings.slice([0..x1, 0..c1, 0..h1, 0..target_batch_size.min(w1)]),
            batch_size,
            3,
        )?;

        // Split the 35 batch size from pyramid encoding back into 5x5+3x3+1x1.
        let chunks = x_pyramid_encoding.split_with_sizes(vec![x0_b, x1_b, x2_b], 0);
        let Some([x0_encodings, x1_encodings, x2_encodings]) = chunks.get(0..3) else {
            return Err(anyhow!("Failed to split x_pyramid_encoding"));
        };

        // 96x96 feature maps by merging 5x5 @ 24x24 patches with overlaps.
        let x0_features = self.merge(x0_encodings.clone(), batch_size, 3)?;
        //  48x84 feature maps by merging 3x3 @ 24x24 patches with overlaps.
        let x1_features = self.merge(x1_encodings.clone(), batch_size, 6)?;
        // 24x24 feature maps.
        let x2_features = x2_encodings.clone();

        // Apply the image encoder model.
        let x_global_features = image_encoder
            .forward(x2_patches, device)
            .map(|res| self.reshape_feature(res.tensor, self.out_size, self.out_size))?;

        // Upsample feature maps.
        let x_latent0_features = self.upsample_latent0.forward(x0_latent_features);
        let x_latent1_features = self.upsample_latent1.forward(x1_latent_features);

        let x0_features_upsampled = self.upsample0.forward(x0_features);
        let x1_features_upsampled = self.upsample1.forward(x1_features);
        let x2_features_upsampled = self.upsample2.forward(x2_features);

        let x_global_features = self.upsample_lowres.forward(x_global_features);

        let x_global_features = self.fuse_lowres.forward(Tensor::cat(
            vec![x2_features_upsampled, x_global_features],
            1,
        ));

        Ok(EncoderOutput {
            x_latent0_features,
            x_latent1_features,
            x0_features: x0_features_upsampled,
            x1_features: x1_features_upsampled,
            x_global_features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Metal;
    use burn::record::{FullPrecisionSettings, Recorder};
    use burn::tensor::TensorData;
    use burn_import::pytorch::PyTorchFileRecorder;
    use ndarray::Array4;

    fn create_encoder_with_weight() -> Encoder<Metal> {
        let device = Default::default();

        let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
            .load(
                "/Users/marcintha/workspace/brioche/butter/encoder_only.pt".into(),
                &device,
            )
            .unwrap();

        let encoder_config = EncoderConfig {
            dims_encoder: vec![256, 512, 1024, 1024],
            patch_encoder_embed_dim: 1024,
            image_encoder_embed_dim: 1024,
            decoder_features: 256,
            out_size: 384 / 16,
        };

        let encoder = Encoder::<Metal>::new(encoder_config, &device).load_record(record);

        encoder
    }

    #[test]
    fn expect_encoder_to_generate_something() {
        let device = Default::default();
        // Create a patch encoder. The weight is loaded from the .onnx.data file automatically.
        let patch_encoder = PatchVitModel::new(
            "/Users/marcintha/workspace/brioche/butter/depthpro_vit_patch.onnx",
            4,
        )
        .unwrap();

        // Create the image encoder as well
        // @TODO change the input to take AsRef<path>
        let image_encoder = CommonVitModel::new(
            "/Users/marcintha/workspace/brioche/butter/depthpro_vit_image.onnx".into(),
            4,
        )
        .unwrap();

        // Create x
        let x_matrix: Array4<f32> =
            ndarray_npy::read_npy("testdata/tensors_data/fov/x.npy").unwrap();
        let (x_data, _) = x_matrix.into_raw_vec_and_offset();
        let x: Tensor<Metal, 4> =
            Tensor::from_data(TensorData::new(x_data, [1, 3, 1536, 1536]), &device);

        let encoder = create_encoder_with_weight();
        let res = encoder
            .forward(x, patch_encoder, image_encoder, &device)
            .unwrap();

        assert_eq!(res.x_latent0_features.shape().dims(), [1, 256, 768, 768]);
        assert_eq!(res.x_latent1_features.shape().dims(), [1, 256, 384, 384]);
        assert_eq!(res.x0_features.shape().dims(), [1, 512, 192, 192]);
        assert_eq!(res.x1_features.shape().dims(), [1, 1024, 96, 96]);
        assert_eq!(res.x_global_features.shape().dims(), [1, 1024, 48, 48]);
    }
}
