import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed


# --- These are just dummy classes to match depth-pro structure in order to export the encoders --- #
# /!\ Because these are ViT model, it's pretty hard to import it in Burn models directly. Hence exporting the model + weight to onnx
class DummyDepthProModel(nn.Module):
    def __init__(
        self,
        fov_encoder: nn.Module,
        patch_encoder: nn.Module,
        image_encoder: nn.Module,
    ):
        super().__init__()

        self.fov = FovEncoder(fov_encoder)
        self.patch_encoder = PatchEncoder(patch_encoder, block0=5, block1=11)
        self.image_encoder = ImageEncoder(image_encoder)


# Dummy model
class FovEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.encoder = model


# Dummy model
class ImageEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.encoder = model


# Dummy model
class PatchEncoder(nn.Module):
    def __init__(self, model: nn.Module, block0: int, block1: int):
        super().__init__()

        self.encoder = model
        self.block0 = block0
        self.block1 = block1
        self.hook0: torch.Tensor | None = None
        self.hook1: torch.Tensor | None = None

    def forward(self, x):
        # Add the hooks to the model when running the forward pass method to the encoder.
        # /!\ Forward hooks are not supported by burn. So we'll add a forward hook and then returns the output of the forward hooks to the "output_names" of the encoder along with output in the "dynamic_axes".
        self.encoder.blocks[self.block0].register_forward_hook(self.hook_fn0)
        self.encoder.blocks[self.block1].register_forward_hook(self.hook_fn1)

        # make a forward passthrough of the encoder
        final_output = self.encoder(x)

        return tuple([final_output, self.hook0, self.hook1])

    def hook_fn0(self, module, input, output):
        self.hook0 = output

    def hook_fn1(self, module, input, output):
        self.hook1 = output


# --- Provided by depth-pro ---
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
