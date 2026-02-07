# Brioche 🍞

<p align="center">
  <img src="brioche.png" width="45%" />
</p>

Brioche is a Rust implementation of the [ml-depth-pro](https://github.com/apple/ml-depth-pro#9efe5c1def37a26c5367a71df664b18e1306c708) repository which re-implements the depth-pro neural network model. It uses the [ONNX](https://onnx.ai/) to load the underlying [vit_large_patch14_dinov2](https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m) model and weights.

## Output example below

<p align="center">
  <img src="./assets/input.jpg" width="45%" />
  <img src="test.jpg" width="45%" />
</p>

## Requirements

- You'll need to have the latest version of [Rust](https://www.rust-lang.org/tools/install) installed.
- You'll need to have the [uv package manager](https://docs.astral.sh/uv/) installed. Once you have it installed, you can install the required dependencies by running the following command in the **butter** folder:

```sh
uv sync
```

1. Download the depth-pro checkpoint file by running the following command in the **butter** folder: 

```sh
uv run vit_exporter.py --download-checkpoint --checkpoint-path ./depth_pro.pt
```

2. Export the ONNX model & weights by running the following script:

```sh
uv run vit_exporter.py --fov --checkpoint-path ./depth_pro.pt
uv run vit_exporter.py --patch --checkpoint-path ./depth_pro.pt
uv run vit_exporter.py --image --checkpoint-path ./depth_pro.pt
```

3. Export the weights for the network model by running the following command:

```sh
uv run state_exporter.py --fov --checkpoint-path ./depth_pro.pt
uv run state_exporter.py --encoder --checkpoint-path ./depth_pro.pt
uv run state_exporter.py --decoder --checkpoint-path ./depth_pro.pt
uv run state_exporter.py --head --checkpoint-path ./depth_pro.pt
```

4. Once you have all these files you should be able to run the sample using this command from the **root** folder of this repository.

```sh
cargo run --release --example sample_metal --features="metal"
```

## Note ⚠️

This project is not production ready. It's a work that is done for educational purposes. There's currently an overhead when converting a CPU tensor from ONNX to GPU tensor for Burn which is not ideal for production use.
