import onnx
from onnxconverter_common import float16

# fov
fov = onnx.load("depthpro_vit_fov.onnx")
fov_fp16 = float16.convert_float_to_float16(fov)
onnx.save(fov_fp16, "depthpro_vit_fov_f16.onnx")

# patch
patch = onnx.load("depthpro_vit_patch.onnx")
patch_fp16 = float16.convert_float_to_float16(patch)
onnx.save(patch_fp16, "depthpro_vit_patch_f16.onnx")

# image
image = onnx.load("depthpro_vit_image.onnx")
image_fp16 = float16.convert_float_to_float16(image)
onnx.save(image_fp16, "depthpro_vit_image_f16.onnx")
