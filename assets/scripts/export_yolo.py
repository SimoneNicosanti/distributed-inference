from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from export_common import (
    DEFAULT_OUTPUT_DIR,
    ImageCalibrationReader,
    ensure_directory,
    finalize_model,
    get_onnx_input_name,
    resolve_export_opset,
    resolve_opset,
    sanitize_model_name,
    yolo_preprocessor,
)

TASK_SUFFIX = {
    "detect": "",
    "segment": "-seg",
    "pose": "-pose",
    "obb": "-obb",
    "classify": "-cls",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an Ultralytics YOLO model to ONNX"
    )
    parser.add_argument(
        "--model", default=None, help="Explicit .pt path or official checkpoint name"
    )
    parser.add_argument(
        "--version", default="yolo11", help="For example yolo11 or yolo26"
    )
    parser.add_argument("--size", choices=["n", "s", "m", "l", "x"], default="n")
    parser.add_argument(
        "--task",
        choices=["detect", "segment", "pose", "obb", "classify"],
        default="detect",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--simplify", action=argparse.BooleanOptionalAction, default=True
    )
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
        raise ValueError("--calibration-dir is required for static INT8 YOLO export")

    checkpoint = args.model or f"{args.version}{args.size}{TASK_SUFFIX[args.task]}.pt"
    opset = resolve_opset(args.opset)
    export_opset = resolve_export_opset(opset)
    output_dir = ensure_directory(args.output_dir.resolve())

    from ultralytics import YOLO

    yolo = YOLO(checkpoint)
    exported_path = Path(
        yolo.export(
            format="onnx",
            imgsz=(args.height, args.width),
            batch=args.batch_size,
            dynamic=args.dynamic,
            simplify=args.simplify,
            opset=export_opset,
            device="cpu",
            nms=False,
        )
    ).resolve()

    stem = f"{sanitize_model_name(checkpoint)}_b{args.batch_size}_{args.precision}"
    raw_path = output_dir / f"{stem}_raw.onnx"
    final_path = output_dir / f"{stem}.onnx"
    shutil.move(str(exported_path), raw_path)

    calibration_reader = None
    if args.precision == "int8":
        calibration_reader = ImageCalibrationReader(
            args.calibration_dir,
            get_onnx_input_name(raw_path),
            yolo_preprocessor(
                args.height,
                args.width,
                1 if args.dynamic else args.batch_size,
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
