#![recursion_limit = "256"]

use brioche::four::Four;
use burn::backend::Metal;

fn main() {
    println!("Running little cat");
    //let device = Default::default();
    let four = Four::<Metal>::new(
        "/Users/marcintha/workspace/brioche/butter/depthpro_vit_fov.onnx",
        "/Users/marcintha/workspace/brioche/butter/depthpro_vit_patch.onnx",
        "/Users/marcintha/workspace/brioche/butter/depthpro_vit_image.onnx",
        3,
        "/Users/marcintha/workspace/brioche/butter/fov_only.pt",
        "/Users/marcintha/workspace/brioche/butter/encoder_only.pt",
        "/Users/marcintha/workspace/brioche/butter/decoder_only.pt",
        "/Users/marcintha/workspace/brioche/butter/head.pt",
    )
    .unwrap();

    let res = four.run("/Users/marcintha/workspace/ml-depth-pro/data/example.jpg");
    dbg!(res);
    println!("Ended");
}
