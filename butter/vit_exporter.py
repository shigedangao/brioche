import argparse
import os

import onnx
import timm
import torch
import torch.nn as nn
import wget
from onnxconverter_common import float16
from torch.export import Dim

from depth_pro_dummy_model import DummyDepthProModel, resize_model, resize_patch_embed

parser = argparse.ArgumentParser(description="Export ViT model to ONNX")
parser.add_argument("--checkpoint-path", type=str, required=True)
parser.add_argument("--download-checkpoint", action=argparse.BooleanOptionalAction)
parser.add_argument("--half", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

if args.download_checkpoint:
    print("Downloading checkpoint...")
    wget.download(
        "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt",
        f"{args.checkpoint_path}",
    )
    print("Checkpoint downloaded successfully.")
    exit()

batch = Dim("batch", min=1)  # or min=1, max=64 if you want guards
height = Dim("height", min=384)  # you can set min/max; keep 384 if fixed
width = Dim("width", min=384)

config = {
    "in_chans": 3,
    "embed_dim": 1024,
    "encoder_feature_layer_ids": [5, 11, 17, 23],
    "encoder_feature_dims": [256, 512, 1024, 1024],
    "img_size": 384,
    "patch_size": 16,
    "timm_preset": "vit_large_patch14_dinov2",
    "timm_img_size": 518,
    "timm_patch_size": 14,
}


# Create a minimal ViT model for ONNX export
def create_minimal_vit_to_onnx() -> nn.Module:
    vit = timm.create_model(
        "vit_large_patch14_dinov2", pretrained=True, dynamic_img_size=True
    )

    vit_model = nn.Module()
    vit_model.hooks = config["encoder_feature_layer_ids"]
    vit_model.model = vit
    vit_model.features = config["encoder_feature_dims"]
    vit_model.vit_features = config["embed_dim"]
    vit_model.model.start_index = 1
    vit_model.model.patch_size = vit_model.model.patch_embed.patch_size
    vit_model.model.is_vit = True
    vit_model.model.forward = vit_model.model.forward_features

    model = resize_patch_embed(vit_model.model, new_patch_size=(16, 16))
    model = resize_model(model, img_size=(384, 384))

    return model


# Create a dummy DepthPro model wrapping the ViT encoder for ONNX export
model = DummyDepthProModel(
    fov_encoder=create_minimal_vit_to_onnx(),
    patch_encoder=create_minimal_vit_to_onnx(),
    image_encoder=create_minimal_vit_to_onnx(),
).to("cpu")

# Import the weight for the ViT encoder into the DepthPro model (.pt file)
state_dict = torch.load(args.checkpoint_path, map_location="cpu")

# Remap the state dictionary keys to match the DepthPro model structure
remapped_state = {}
for key, value in state_dict.items():
    if key.startswith("encoder.patch_encoder."):
        # encoder.patch_encoder.X -> patch_encoder.encoder.X
        new_key = key.replace("encoder.patch_encoder.", "patch_encoder.encoder.")
        remapped_state[new_key] = value
    elif key.startswith("encoder.image_encoder."):
        # encoder.image_encoder.X -> image_encoder.encoder.X
        new_key = key.replace("encoder.image_encoder.", "image_encoder.encoder.")
        remapped_state[new_key] = value
    elif key.startswith("fov.encoder."):
        # fov.encoder.X -> fov.encoder.X (should match)
        new_key = key.replace("fov.encoder.", "fov.encoder.")
        remapped_state[new_key] = value

# Load the weights into the model
model.load_state_dict(remapped_state, strict=False)

dummy_inputs = {
    "patch": torch.randn(35, 3, 384, 384),
    "image": torch.randn(1, 3, 384, 384),
    "fov": torch.randn(1, 3, 384, 384),
}

output_names = {
    "patch": ["final_output", "hooks0", "hooks1"],
    "image": ["tokens"],
    "fov": ["tokens"],
}

models_ref = {
    "patch": model.patch_encoder,
    "image": model.image_encoder.encoder,
    "fov": model.fov.encoder,
}

# Create the directory if it doesn't exist
if not os.path.exists("./onnx_model"):
    os.makedirs("./onnx_model")

for config in ["patch", "image", "fov"]:
    model_to_export = models_ref[config]
    dummy_input = dummy_inputs[config]
    output_name = output_names[config]
    filename = f"depthpro_vit_{config}"

    torch.onnx.export(
        model_to_export,
        dummy_input,
        f"./onnx_model/{filename}.onnx",
        opset_version=21,
        input_names=["x"],
        output_names=output_name,
        dynamo=True,
        dynamic_shapes={"x": {0: Dim("batch", min=1)}},
    )

    # Load the weights and re-export the model with the updated weights
    m = onnx.load(f"./onnx_model/{filename}.onnx")
    onnx.save_model(
        m,
        f"./onnx_model/{filename}.onnx",
        save_as_external_data=False,
    )

    if args.half:
        model_f32 = onnx.load(f"./onnx_model/{filename}.onnx")
        model_fp16 = float16.convert_float_to_float16(model_f32)
        onnx.save(model_fp16, f"./onnx_model/{filename}_half.onnx")
