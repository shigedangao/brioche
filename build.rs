#[cfg(feature = "burn_onnx")]
use burn_onnx::ModelGen;

fn main() {
    #[cfg(feature = "burn_onnx")]
    ModelGen::new()
        .input("./butter/onnx_model/depthpro_vit_patch.onnx")
        .input("./butter/onnx_model/depthpro_vit_image.onnx")
        .input("./butter/onnx_model/depthpro_vit_fov.onnx")
        .out_dir("model/")
        .run_from_script();
}
