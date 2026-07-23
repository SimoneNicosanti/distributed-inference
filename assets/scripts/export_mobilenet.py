from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.models import get_model, get_model_weights

from export_common import (
    DEFAULT_OUTPUT_DIR,
    ImageCalibrationReader,
    ensure_directory,
    export_torch_model,
    finalize_model,
    get_onnx_input_name,
    imagenet_preprocessor,
    resolve_export_opset,
    resolve_opset,
    sanitize_model_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a torchvision MobileNet to ONNX"
    )
    parser.add_argument("--model", default="mobilenet_v3_large")
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--dynamic-spatial", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--calibration-samples", type=int, default=100)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.precision == "int8" and args.calibration_dir is None:
        raise ValueError(
            "--calibration-dir is required for static INT8 MobileNet export"
        )

    opset = resolve_opset(args.opset)
    export_opset = resolve_export_opset(opset)
    output_dir = ensure_directory(args.output_dir.resolve())
    weights_enum = get_model_weights(args.model)
    weights = (
        weights_enum.DEFAULT
        if args.weights == "DEFAULT"
        else weights_enum[args.weights]
    )
    model = get_model(args.model, weights=weights).eval()

    dummy = torch.randn(args.batch_size, 3, args.height, args.width)
    dynamic_axes: dict[str, dict[int, str]] = {}
    if args.dynamic_batch:
        dynamic_axes.setdefault("pixel_values", {})[0] = "batch_size"
        dynamic_axes.setdefault("logits", {})[0] = "batch_size"
    if args.dynamic_spatial:
        dynamic_axes.setdefault("pixel_values", {}).update({2: "height", 3: "width"})

    stem = f"{sanitize_model_name(args.model)}_b{args.batch_size}_{args.precision}"
    raw_path = output_dir / f"{stem}_raw.onnx"
    final_path = output_dir / f"{stem}.onnx"

    export_torch_model(
        model,
        (dummy,),
        raw_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        opset=export_opset,
        dynamic_axes=dynamic_axes or None,
    )

    calibration_reader = None
    if args.precision == "int8":
        calibration_reader = ImageCalibrationReader(
            args.calibration_dir,
            get_onnx_input_name(raw_path),
            imagenet_preprocessor(
                args.height,
                args.width,
                1 if args.dynamic_batch else args.batch_size,
            ),
            args.calibration_samples,
        )

    result = finalize_model(
        raw_path,
        final_path,
        opset=opset,
        precision=args.precision,
        calibration_reader=calibration_reader,
        keep_intermediate=args.keep_intermediate,
        is_transformer=False,
    )
    print(result)


if __name__ == "__main__":
    main()
