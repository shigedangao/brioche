use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("./butter/onnx_model/depthpro_vit_fov.onnx")
        .out_dir("model/")
        .run_from_script();

    ModelGen::new()
        .input("./butter/onnx_model/depthpro_vit_image.onnx")
        .out_dir("model/")
        .run_from_script();
}
