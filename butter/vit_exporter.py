import onnxruntime as ort
import timm
import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed
from torch.export import Dim

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


# --- These are just dummy classes to match depth-pro structure in order to export the encoders --- #
# /!\ Because these are ViT model, it's pretty hard to import it in Burn models directly. Hence exporting the model + weight to onnx
class DummyDepthProModel(nn.Module):
    def __init__(
        self,
        fov_encoder: nn.Module,
    ):
        super().__init__()

        self.fov = FovEncoder(fov_encoder)


class FovEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.encoder = model


# --- Provided by depth-pro (yours, unchanged) ---
def resize_patch_embed(model: nn.Module, new_patch_size=(16, 16)) -> nn.Module:
    if hasattr(model, "patch_embed"):
        old_patch_size = model.patch_embed.patch_size
        if new_patch_size != old_patch_size:
            w = model.patch_embed.proj.weight
            b = model.patch_embed.proj.bias
            use_bias = b is not None
            _, _, h, w0 = w.shape

            new_w = torch.nn.functional.interpolate(
                w,
                size=[new_patch_size[0], new_patch_size[1]],
                mode="bicubic",
                align_corners=False,
            )
            new_w = new_w * (h / new_patch_size[0]) * (w0 / new_patch_size[1])

            model.patch_embed.proj = nn.Conv2d(
                in_channels=model.patch_embed.proj.in_channels,
                out_channels=model.patch_embed.proj.out_channels,
                kernel_size=new_patch_size,
                stride=new_patch_size,
                bias=use_bias,
            )
            if use_bias:
                model.patch_embed.proj.bias = b
            model.patch_embed.proj.weight = nn.Parameter(new_w)

            model.patch_size = new_patch_size
            model.patch_embed.patch_size = new_patch_size
            model.patch_embed.img_size = (
                int(
                    model.patch_embed.img_size[0]
                    * new_patch_size[0]
                    / old_patch_size[0]
                ),
                int(
                    model.patch_embed.img_size[1]
                    * new_patch_size[1]
                    / old_patch_size[1]
                ),
            )
    return model


def resize_model(model: nn.Module, img_size: tuple[int, int]) -> nn.Module:
    patch_size = model.patch_embed.patch_size
    model.patch_embed.img_size = img_size
    grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
    model.patch_embed.grid_size = grid_size

    pos_embed = resample_abs_pos_embed(
        model.pos_embed,
        grid_size,
        num_prefix_tokens=(
            0 if getattr(model, "no_embed_class", False) else model.num_prefix_tokens
        ),
    )
    model.pos_embed = nn.Parameter(pos_embed)
    return model


def create_minimal_vit_to_onnx() -> nn.Module:
    vit = timm.create_model(
        "vit_large_patch14_dinov2", pretrained=True, dynamic_img_size=True
    )

    # kill classifier head just in case
    if hasattr(vit, "reset_classifier"):
        vit.reset_classifier(0)
    elif hasattr(vit, "head"):
        vit.head = nn.Identity()

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


# Create a bare ViT model suitable for DepthPro after resizing the patch embedding and position embeddings
base_model = create_minimal_vit_to_onnx()

# Create a dummy DepthPro model wrapping the ViT encoder for ONNX export
model = DummyDepthProModel(fov_encoder=base_model).to("cpu")

# Import the weight for the ViT encoder into the DepthPro model (.pt file)
state_dict = torch.load(
    "/Users/marcintha/workspace/ml-depth-pro/checkpoints/depth_pro.pt",
    map_location="cpu",
)

# Load the weights into the model
model.load_state_dict(state_dict, strict=False)

# choose a dummy input tensor but which match the ViT encoder (fov, patch_encoder, image_encoder) that will be export to onnx
dummy = torch.randn(1, 3, 384, 384)

torch.onnx.export(
    model.fov.encoder,
    dummy,
    "depthpro_vit_fov.onnx",
    opset_version=21,
    input_names=["x"],
    output_names=["tokens"],
    dynamo=True,
    dynamic_shapes={"x": {0: Dim("batch", min=1)}},
)
