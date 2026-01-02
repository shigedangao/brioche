import torch

# Load the pt file and re-export the field that is needed
checkpoint = torch.load(
    "/Users/marcintha/workspace/ml-depth-pro/checkpoints/depth_pro.pt"
)

if "state_dict" in checkpoint:
    state = checkpoint["state_dict"]
else:
    state = checkpoint

rename_fov_keys = {
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

filtered_fov_state = {
    rename_fov_keys[k]: v for k, v in state.items() if k in rename_fov_keys
}

torch.save(filtered_fov_state, "fov_only.pt")
