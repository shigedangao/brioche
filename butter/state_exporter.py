import argparse

import torch

parser = argparse.ArgumentParser(description="Export state of the depth pro model")
parser.add_argument("--fov", type=bool, required=False)
parser.add_argument("--encoder", type=bool, required=False)

args = parser.parse_args()

# Load the pt file and re-export the field that is needed
checkpoint = torch.load(
    "/Users/marcintha/workspace/ml-depth-pro/checkpoints/depth_pro.pt"
)

if "state_dict" in checkpoint:
    state = checkpoint["state_dict"]
else:
    state = checkpoint

rename_keys = {
    "fov.downsample.0.weight": "downsample.conv.weight",
    "fov.downsample.0.bias": "downsample.conv.bias",
    "fov.head.0.weight": "head.conv64.weight",
    "fov.head.0.bias": "head.conv64.bias",
    "fov.head.2.weight": "head.conv32.weight",
    "fov.head.2.bias": "head.conv32.bias",
    "fov.head.4.weight": "head.conv16.weight",
    "fov.head.4.bias": "head.conv16.bias",
    "fov.encoder.1.bias": "encoder.linear.bias",
    "fov.encoder.1.weight": "encoder.linear.weight",
}
export_name = "fov_only.pt"

if args.encoder:
    rename_keys = {
        "encoder.upsample_latent0.0.weight": "upsample_latent0.conv2d.weight",
        "encoder.upsample_latent0.1.weight": "upsample_latent0.blocks.0.weight",
        "encoder.upsample_latent0.2.weight": "upsample_latent0.blocks.1.weight",
        "encoder.upsample_latent0.3.weight": "upsample_latent0.blocks.2.weight",
        "encoder.upsample_latent1.0.weight": "upsample_latent1.conv2d.weight",
        "encoder.upsample_latent1.1.weight": "upsample_latent1.blocks.0.weight",
        "encoder.upsample_latent1.2.weight": "upsample_latent1.blocks.1.weight",
        "encoder.upsample0.0.weight": "upsample0.conv2d.weight",
        "encoder.upsample0.1.weight": "upsample0.blocks.0.weight",
        "encoder.upsample1.0.weight": "upsample1.conv2d.weight",
        "encoder.upsample1.1.weight": "upsample1.blocks.0.weight",
        "encoder.upsample2.0.weight": "upsample2.conv2d.weight",
        "encoder.upsample2.1.weight": "upsample2.blocks.0.weight",
        "encoder.upsample_lowres.weight": "upsample_lowres.weight",
        "encoder.upsample_lowres.bias": "upsample_lowres.bias",
        "encoder.fuse_lowres.weight": "fuse_lowres.weight",
        "encoder.fuse_lowres.bias": "fuse_lowres.bias",
    }
    export_name = "encoder_only.pt"

filtered_fov_state = {rename_keys[k]: v for k, v in state.items() if k in rename_keys}

torch.save(filtered_fov_state, export_name)
