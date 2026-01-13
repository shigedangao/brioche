use brioche::four::Four;
use burn::backend::Wgpu;

fn main() {
    //let device = Default::default();
    Four::<Wgpu>::new(
        "/Users/marcintha/workspace/brioche/butter/depthpro_vit_fov.onnx",
        "/Users/marcintha/workspace/brioche/butter/depthpro_vit_patch.onnx",
        "Users/marcintha/workspace/brioche/butter/depthpro_vit_image.onnx",
        4,
        "/Users/marcintha/workspace/brioche/butter/fov_only.pt",
        "/Users/marcintha/workspace/brioche/butter/encoder_only.pt",
        "/Users/marcintha/workspace/brioche/butter/decoder_only.pt",
    )
    .unwrap();
}
