#![recursion_limit = "256"]
use std::time::Instant;

use brioche::four::Four;
use burn::backend::Metal;
use burn::tensor::f16;

fn main() {
    let t = Instant::now();

    println!("Running sample of loaf 🍞");
    //let device = Default::default();
    let four = Four::<Metal>::new(
        "./butter/onnx_model/depthpro_vit_fov_f16.onnx",
        "./butter/onnx_model/depthpro_vit_patch_f16.onnx",
        "./butter/onnx_model/depthpro_vit_image_f16.onnx",
        4,
        "./butter/weights/fov_only.pt",
        "./butter/weights/encoder_only.pt",
        "./butter/weights/decoder_only.pt",
        "./butter/weights/head.pt",
    )
    .unwrap();

    println!("model initialized at {:?}", t.elapsed());

    let (img_buffer, focallength_px) = four
        .run::<_, f16>("./input.jpg", true)
        .expect("Expect to have generate the image");

    img_buffer
        .save("test.jpg")
        .expect("Expect to save the image");

    dbg!(focallength_px);

    println!("Image has been generated in {:?}", t.elapsed());
}
