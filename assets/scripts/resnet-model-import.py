from pathlib import Path

import onnx
import torch
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet152_Weights, resnet152


output_path = Path("../models/resnet50.onnx")

# Download and load pretrained ImageNet weights.
# weights = ResNet50_Weights.DEFAULT
# model = resnet50(weights=weights)
weights = ResNet152_Weights.DEFAULT
model = resnet152(weights=weights)
model.eval()

# ResNet-50 conventionally accepts images shaped:
# (batch, channels, height, width)
example_input = torch.randn(
    1,
    3,
    224,
    224,
    dtype=torch.float32,
)

with torch.inference_mode():
    onnx_program = torch.onnx.export(
        model,
        (example_input,),
        dynamo=True,
        input_names=["images"],
        output_names=["logits"],
    )

onnx_program.save(output_path)

# Validate the exported protobuf.
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)

print(f"Saved valid ONNX model to {output_path}")
