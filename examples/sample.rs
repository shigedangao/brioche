#![recursion_limit = "256"]
use brioche::four::{Four, FourConfig};
#[cfg(feature = "f16")]
use burn::tensor::f16;
use clap::Parser;
use spinners::{Spinner, Spinners};
use std::sync::LazyLock;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "metal")]
use burn::backend::Metal as Backend;

#[cfg(feature = "cuda")]
use burn::backend::Cuda as Backend;

#[cfg(feature = "rocm")]
use burn::backend::Rocm as Backend;

static CONFIG: LazyLock<FourConfig<&'static str>> = cfg_select! {
    feature = "f32" => LazyLock::new(|| FourConfig {
        patch_vit_path: "./butter/onnx_model/depthpro_vit_patch.onnx",
        image_vit_path: "./butter/onnx_model/depthpro_vit_image.onnx",
        fov_vit_path: "./butter/onnx_model/depthpro_vit_fov.onnx",
        fov_weight_path: "./butter/weights/fov_only.pt",
        encoder_weight_path: "./butter/weights/encoder_only.pt",
        decoder_weight_path: "./butter/weights/decoder_only.pt",
        head_weight_path: "./butter/weights/head.pt",
        vit_thread_nb: 8,
    }),
    feature = "f16" => LazyLock::new(|| FourConfig {
        patch_vit_path: "./butter/onnx_model/depthpro_vit_patch_f16.onnx",
        image_vit_path: "./butter/onnx_model/depthpro_vit_image_f16.onnx",
        fov_vit_path: "./butter/onnx_model/depthpro_vit_fov_f16.onnx",
        fov_weight_path: "./butter/weights/fov_only.pt",
        encoder_weight_path: "./butter/weights/encoder_only.pt",
        decoder_weight_path: "./butter/weights/decoder_only.pt",
        head_weight_path: "./butter/weights/head.pt",
        vit_thread_nb: 8,
    }),
};

#[cfg(feature = "f32")]
static CONFIG_QUANTIZE: LazyLock<FourConfig<&'static str>> = LazyLock::new(|| FourConfig {
    patch_vit_path: "./butter/onnx_model/depthpro_vit_patch_quantize.onnx",
    image_vit_path: "./butter/onnx_model/depthpro_vit_image_quantize.onnx",
    fov_vit_path: "./butter/onnx_model/depthpro_vit_fov_quantize.onnx",
    fov_weight_path: "./butter/weights/fov_only.pt",
    encoder_weight_path: "./butter/weights/encoder_only.pt",
    decoder_weight_path: "./butter/weights/decoder_only.pt",
    head_weight_path: "./butter/weights/head.pt",
    vit_thread_nb: 8,
});

#[derive(Parser, Debug, Clone)]
#[command(version, about)]
struct Cli {
    #[arg(long, default_value = "./assets/input.jpg")]
    source: String,

    #[arg(long, default_value = "false")]
    quantize: bool,
}

fn main() {
    let cli = Cli::parse();
    let t = Instant::now();

    let config = match cli.quantize {
        true => *CONFIG_QUANTIZE,
        false => *CONFIG,
    };

    println!("Running sample of loaf 🍞");
    let four = Four::<Backend>::new::<&'static str>(config).unwrap();

    println!("model initialized at {:?}", t.elapsed());

    // start spinner
    let mut sp = Spinner::new(Spinners::Dots8, "Generating image...".into());

    #[cfg(feature = "f32")]
    let (img_buffer, _focallength_px) = four
        .run::<_, f32>(cli.source, false)
        .expect("Expect to have generate the image");

    #[cfg(feature = "f16")]
    println!("Run model with F16");
    #[cfg(feature = "f16")]
    let (img_buffer, _focallength_px) = four
        .run::<_, f16>(cli.source, true)
        .expect("Expect to have generate the image");

    let now = SystemTime::now();
    let timestamp = now
        .duration_since(UNIX_EPOCH)
        .expect("Expect to get timestamp");

    img_buffer
        .save(format!("./test-{:?}.jpg", timestamp))
        .expect("Expect to save the image");

    // stop spinner
    sp.stop_with_message(format!("Image has been generated in {:?}", t.elapsed()));
}
