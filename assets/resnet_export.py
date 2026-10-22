from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from torchvision.models import ResNet50_Weights, resnet50


def main() -> None:
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = output_dir / "resnet50.onnx"
    optimized_path = output_dir / "resnet50_optimized.onnx"

    # 1. Load pretrained ResNet50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()

    # 2. Export to ONNX
    example_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (example_input,),
            str(original_path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=18,
            dynamo=False,
            external_data=False,
            dynamic_axes={
                "images": {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
        )

    # 3. Validate exported model
    onnx.checker.check_model(str(original_path))

    # 4. Optimize graph with ONNX Runtime
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(optimized_path)

    session = ort.InferenceSession(
        str(original_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    # 5. Validate optimized model
    onnx.checker.check_model(str(optimized_path))

    # 6. Check dynamic batch execution
    images = torch.randn(2, 3, 224, 224).numpy()
    outputs = session.run(None, {"images": images})

    print(f"Original:  {original_path}")
    print(f"Optimized: {optimized_path}")
    print(f"Output shape: {outputs[0].shape}")


if __name__ == "__main__":
    main()
