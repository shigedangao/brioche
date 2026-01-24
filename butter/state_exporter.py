import argparse

import torch

parser = argparse.ArgumentParser(description="Export state of the depth pro model")
parser.add_argument("--fov", type=bool, required=False)
parser.add_argument("--encoder", type=bool, required=False)
parser.add_argument("--decoder", type=bool, required=False)
parser.add_argument("--head", type=bool, required=False)

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
elif args.decoder:
    rename_keys = {
        # Start from convs.0 as the dimensions are equal @todo to change.
        "decoder.convs.1.weight": "convs.1.weight",
        "decoder.convs.2.weight": "convs.2.weight",
        "decoder.convs.3.weight": "convs.3.weight",
        "decoder.convs.4.weight": "convs.4.weight",
        "decoder.fusions.0.resnet1.residual.1.weight": "fusions.0.resnet1.sequential.0.conv2d.weight",
        "decoder.fusions.0.resnet1.residual.1.bias": "fusions.0.resnet1.sequential.0.conv2d.bias",
        "decoder.fusions.0.resnet1.residual.3.weight": "fusions.0.resnet1.sequential.1.conv2d.weight",
        "decoder.fusions.0.resnet1.residual.3.bias": "fusions.0.resnet1.sequential.1.conv2d.bias",
        "decoder.fusions.0.resnet2.residual.1.weight": "fusions.0.resnet2.sequential.0.conv2d.weight",
        "decoder.fusions.0.resnet2.residual.1.bias": "fusions.0.resnet2.sequential.0.conv2d.bias",
        "decoder.fusions.0.resnet2.residual.3.weight": "fusions.0.resnet2.sequential.1.conv2d.weight",
        "decoder.fusions.0.resnet2.residual.3.bias": "fusions.0.resnet2.sequential.1.conv2d.bias",
        "decoder.fusions.0.out_conv.weight": "fusions.0.outconv.weight",
        "decoder.fusions.0.out_conv.bias": "fusions.0.outconv.bias",
        "decoder.fusions.1.resnet1.residual.1.weight": "fusions.1.resnet1.sequential.0.conv2d.weight",
        "decoder.fusions.1.resnet1.residual.1.bias": "fusions.1.resnet1.sequential.0.conv2d.bias",
        "decoder.fusions.1.resnet1.residual.3.weight": "fusions.1.resnet1.sequential.1.conv2d.weight",
        "decoder.fusions.1.resnet1.residual.3.bias": "fusions.1.resnet1.sequential.1.conv2d.bias",
        "decoder.fusions.1.resnet2.residual.1.weight": "fusions.1.resnet2.sequential.0.conv2d.weight",
        "decoder.fusions.1.resnet2.residual.1.bias": "fusions.1.resnet2.sequential.0.conv2d.bias",
        "decoder.fusions.1.resnet2.residual.3.weight": "fusions.1.resnet2.sequential.1.conv2d.weight",
        "decoder.fusions.1.resnet2.residual.3.bias": "fusions.1.resnet2.sequential.1.conv2d.bias",
        "decoder.fusions.1.deconv.weight": "fusions.1.deconv.weight",
        "decoder.fusions.1.out_conv.weight": "fusions.1.outconv.weight",
        "decoder.fusions.1.out_conv.bias": "fusions.1.outconv.bias",
        "decoder.fusions.2.resnet1.residual.1.weight": "fusions.2.resnet1.sequential.0.conv2d.weight",
        "decoder.fusions.2.resnet1.residual.1.bias": "fusions.2.resnet1.sequential.0.conv2d.bias",
        "decoder.fusions.2.resnet1.residual.3.weight": "fusions.2.resnet1.sequential.1.conv2d.weight",
        "decoder.fusions.2.resnet1.residual.3.bias": "fusions.2.resnet1.sequential.1.conv2d.bias",
        "decoder.fusions.2.resnet2.residual.1.weight": "fusions.2.resnet2.sequential.0.conv2d.weight",
        "decoder.fusions.2.resnet2.residual.1.bias": "fusions.2.resnet2.sequential.0.conv2d.bias",
        "decoder.fusions.2.resnet2.residual.3.weight": "fusions.2.resnet2.sequential.1.conv2d.weight",
        "decoder.fusions.2.resnet2.residual.3.bias": "fusions.2.resnet2.sequential.1.conv2d.bias",
        "decoder.fusions.2.deconv.weight": "fusions.2.deconv.weight",
        "decoder.fusions.2.out_conv.weight": "fusions.2.outconv.weight",
        "decoder.fusions.2.out_conv.bias": "fusions.2.outconv.bias",
        "decoder.fusions.3.resnet1.residual.1.weight": "fusions.3.resnet1.sequential.0.conv2d.weight",
        "decoder.fusions.3.resnet1.residual.1.bias": "fusions.3.resnet1.sequential.0.conv2d.bias",
        "decoder.fusions.3.resnet1.residual.3.weight": "fusions.3.resnet1.sequential.1.conv2d.weight",
        "decoder.fusions.3.resnet1.residual.3.bias": "fusions.3.resnet1.sequential.1.conv2d.bias",
        "decoder.fusions.3.resnet2.residual.1.weight": "fusions.3.resnet2.sequential.0.conv2d.weight",
        "decoder.fusions.3.resnet2.residual.1.bias": "fusions.3.resnet2.sequential.0.conv2d.bias",
        "decoder.fusions.3.resnet2.residual.3.weight": "fusions.3.resnet2.sequential.1.conv2d.weight",
        "decoder.fusions.3.resnet2.residual.3.bias": "fusions.3.resnet2.sequential.1.conv2d.bias",
        "decoder.fusions.3.deconv.weight": "fusions.3.deconv.weight",
        "decoder.fusions.3.out_conv.weight": "fusions.3.outconv.weight",
        "decoder.fusions.3.out_conv.bias": "fusions.3.outconv.bias",
        "decoder.fusions.4.resnet1.residual.1.weight": "fusions.4.resnet1.sequential.0.conv2d.weight",
        "decoder.fusions.4.resnet1.residual.1.bias": "fusions.4.resnet1.sequential.0.conv2d.bias",
        "decoder.fusions.4.resnet1.residual.3.weight": "fusions.4.resnet1.sequential.1.conv2d.weight",
        "decoder.fusions.4.resnet1.residual.3.bias": "fusions.4.resnet1.sequential.1.conv2d.bias",
        "decoder.fusions.4.resnet2.residual.1.weight": "fusions.4.resnet2.sequential.0.conv2d.weight",
        "decoder.fusions.4.resnet2.residual.1.bias": "fusions.4.resnet2.sequential.0.conv2d.bias",
        "decoder.fusions.4.resnet2.residual.3.weight": "fusions.4.resnet2.sequential.1.conv2d.weight",
        "decoder.fusions.4.resnet2.residual.3.bias": "fusions.4.resnet2.sequential.1.conv2d.bias",
        "decoder.fusions.4.deconv.weight": "fusions.4.deconv.weight",
        "decoder.fusions.4.out_conv.weight": "fusions.4.outconv.weight",
        "decoder.fusions.4.out_conv.bias": "fusions.4.outconv.bias",
    }
    export_name = "decoder_only.pt"
elif args.head:
    rename_keys = {
        "head.0.weight": "conv2d0.weight",
        "head.0.bias": "conv2d0.bias",
        "head.1.weight": "conv_transpose2d.weight",
        "head.1.bias": "conv_transpose2d.bias",
        "head.2.weight": "conv2d1.weight",
        "head.2.bias": "conv2d1.bias",
        "head.4.weight": "conv2d2.weight",
        "head.4.bias": "conv2d2.bias",
    }
    export_name = "head.pt"

filtered_fov_state = {rename_keys[k]: v for k, v in state.items() if k in rename_keys}

torch.save(filtered_fov_state, export_name)
