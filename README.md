# Brioche 🍞 (WIP)

Brioche is a Rust implementation of the [ml-depth-pro](https://github.com/apple/ml-depth-pro#9efe5c1def37a26c5367a71df664b18e1306c708) repository which re-implements the depth-pro neural network model. It uses the [ONNX](https://onnx.ai/) to load the underlying [vit_large_patch14_dinov2](https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m) model and weights.

## Requirements

1. Download the depth-pro checkpoint file by running the following command: 

```sh
uv run vit_exporter.py --download-checkpoint --checkpoint-path ./depth_pro.pt
```
