#![recursion_limit = "256"]
use brioche::four::Four;
use clap::Parser;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "metal")]
use burn::backend::Metal as Backend;

#[cfg(feature = "cuda")]
use burn::backend::Cuda as Backend;

#[derive(Parser, Debug, Clone)]
#[command(version, about)]
struct Cli {
    #[arg(long, default_value = "./assets/lion.jpg")]
    source: String,
}

fn main() {
    let cli = Cli::parse();
    let t = Instant::now();

    println!("Running sample of loaf 🍞");
    let four = Four::<Backend>::new(
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
        .run::<_, f32>(cli.source, false)
        .expect("Expect to have generate the image");

    let now = SystemTime::now();
    let timestamp = now
        .duration_since(UNIX_EPOCH)
        .expect("Expect to get timestamp");
    img_buffer
        .save(format!("./runs/test-{:?}.jpg", timestamp))
        .expect("Expect to save the image");

    dbg!(focallength_px);

    println!("Image has been generated in {:?}", t.elapsed());
}
