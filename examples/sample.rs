#![recursion_limit = "256"]
use brioche::four::Four;
use burn::backend::Metal;
use std::time::Instant;

fn main() {
    let t = Instant::now();

    println!("Running sample of loaf 🍞");
    //let device = Default::default();
    let four = Four::<Metal>::new(
        "./butter/onnx_model/depthpro_vit_patch.onnx",
        6,
        "./butter/weights/fov_only.pt",
        "./butter/weights/encoder_only.pt",
        "./butter/weights/decoder_only.pt",
        "./butter/weights/head.pt",
    )
    .unwrap();

    println!("model initialized at {:?}", t.elapsed());

    let (img_buffer, focallength_px) = four
        .run::<_, f32>("./input.jpg", false)
        .expect("Expect to have generate the image");

    img_buffer
        .save("test.jpg")
        .expect("Expect to save the image");

    dbg!(focallength_px);

    println!("Image has been generated in {:?}", t.elapsed());
}
