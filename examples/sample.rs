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
        "/Users/marcintha/workspace/brioche/butter/depthpro_vit_fov_f16.onnx",
        "/Users/marcintha/workspace/brioche/butter/depthpro_vit_patch_f16.onnx",
        "/Users/marcintha/workspace/brioche/butter/depthpro_vit_image_f16.onnx",
        3,
        "/Users/marcintha/workspace/brioche/butter/fov_only.pt",
        "/Users/marcintha/workspace/brioche/butter/encoder_only.pt",
        "/Users/marcintha/workspace/brioche/butter/decoder_only.pt",
        "/Users/marcintha/workspace/brioche/butter/head.pt",
    )
    .unwrap();

    println!("model initialized at {:?}", t.elapsed());

    four.run::<_, f16>(
        "/Users/marcintha/workspace/ml-depth-pro/data/example.jpg",
        true,
    )
    .expect("Expect to have generate the image");

    println!("Image has been generated in {:?}", t.elapsed());
}
