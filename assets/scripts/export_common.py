from __future__ import annotations

import re
import shutil
import tempfile
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np

Precision = Literal["fp32", "fp16", "int8"]

DEFAULT_OUTPUT_DIR = (Path(__file__).resolve().parent / "../models").resolve()
DEFAULT_INTERMEDIATE_OPSET = 18
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sanitize_model_name(name: str) -> str:
    """Convert a model identifier or path into a filesystem-safe stem."""
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.replace("/", "_"))


def latest_onnx_opset() -> int:
    import onnx

    return int(onnx.defs.onnx_opset_version()) - 1


def resolve_opset(opset: int | None) -> int:
    return latest_onnx_opset() if opset is None else opset


def resolve_export_opset(target_opset: int) -> int:
    """Use a broadly supported export opset, then convert the final model."""
    return min(target_opset, DEFAULT_INTERMEDIATE_OPSET)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_domain_opset(model: Any) -> int | None:
    for opset_import in model.opset_import:
        if opset_import.domain in {"", "ai.onnx"}:
            return int(opset_import.version)
    return None


def ensure_onnx_opset(model_path: Path, target_opset: int) -> None:
    """Ensure the default ONNX domain uses target_opset and validate the result."""
    import onnx
    from onnx import version_converter

    model = onnx.load(str(model_path), load_external_data=True)
    current_opset = _default_domain_opset(model)
    if current_opset is None:
        raise RuntimeError(f"No default ONNX opset found in {model_path}")

    if current_opset != target_opset:
        model = version_converter.convert_version(model, target_opset)

    onnx.checker.check_model(model)
    temporary_path = model_path.with_suffix(".opset_tmp.onnx")
    onnx.save_model(model, str(temporary_path))
    temporary_path.replace(model_path)


def infer_symbolic_shapes(model_path: Path) -> None:
    import onnx
    from onnxruntime.tools.symbolic_shape_infer import (
        SymbolicShapeInference,
    )

    model = onnx.load(str(model_path))

    inferred_model = SymbolicShapeInference.infer_shapes(
        model,
        int_max=2**31 - 1,
        auto_merge=True,
        guess_output_rank=False,
        verbose=0,
    )

    if inferred_model is None:
        raise Exception("Failed to do symbolic shape infer with ORT")

    onnx.checker.check_model(inferred_model)
    onnx.save(inferred_model, str(model_path))


def infer_static_shapes(model_path: Path) -> None:
    import onnx

    try:
        onnx.shape_inference.infer_shapes_path(
            str(model_path),
            check_type=True,
            strict_mode=False,
            data_prop=True,
        )
    except Exception as exc:
        warnings.warn(
            "Standard ONNX shape inference failed; "
            f"the graph will be preserved: {exc!r}",
            stacklevel=2,
        )


def optimize_onnx_basic(model_path: Path, is_transformer: bool) -> None:
    """Apply only ONNX Runtime BASIC graph optimizations."""

    import onnx
    import onnxruntime as ort

    optimized_path = model_path.with_name(f"{model_path.stem}.basic_tmp.onnx")
    optimized_path.unlink(missing_ok=True)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    session_options.optimized_model_filepath = str(optimized_path)

    if is_transformer:
        ## We need to disable this optimization, otherwise the matching with the
        ## attention pattern recognized by ort is broken
        session_options.add_session_config_entry(
            "optimization.disable_specified_optimizers",
            "MatMulAddFusion",
        )

    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        # La serializzazione avviene durante la creazione della sessione.
        del session

        if not optimized_path.exists():
            raise RuntimeError(
                f"ONNX Runtime did not generate the optimized model: {optimized_path}"
            )

        onnx.checker.check_model(str(optimized_path))
        optimized_path.replace(model_path)

    finally:
        optimized_path.unlink(missing_ok=True)


def export_torch_model(
    model: Any,
    args: tuple[Any, ...],
    output_path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
    opset: int,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    import torch

    model.eval()

    mha_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)

    try:
        with torch.inference_mode():
            torch.onnx.export(
                model,
                args,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
                report=True,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(mha_fastpath_enabled)


def convert_to_fp16(input_path: Path, output_path: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(input_path), load_external_data=True)
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=False,
    )
    onnx.save_model(converted, str(output_path))


def quantize_dynamic_int8(input_path: Path, output_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        per_channel=True,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Attention"],
    )


def quantize_static_int8(
    input_path: Path,
    output_path: Path,
    calibration_reader: Any,
) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def finalize_model(
    raw_model_path: Path,
    final_model_path: Path,
    *,
    opset: int,
    precision: Precision,
    calibration_reader: Any | None = None,
    keep_intermediate: bool = False,
    is_transformer: bool = False,
) -> Path:
    """Convert precision, normalize the opset and apply ORT BASIC optimizations.

    Only provider-independent ONNX Runtime BASIC graph optimizations are applied.
    No transformer-specific optimizer or extended/layout optimization is used.
    """
    import onnx

    onnx.checker.check_model(str(raw_model_path))
    ensure_directory(final_model_path.parent)

    with tempfile.TemporaryDirectory(dir=final_model_path.parent) as temporary_dir:
        precision_path = Path(temporary_dir) / "precision.onnx"

        if precision == "fp32":
            working_path = raw_model_path
        elif precision == "fp16":
            convert_to_fp16(raw_model_path, precision_path)
            working_path = precision_path
        elif precision == "int8":
            if calibration_reader is not None:
                quantize_static_int8(raw_model_path, precision_path, calibration_reader)
            else:
                quantize_dynamic_int8(raw_model_path, precision_path)
            working_path = precision_path
        else:
            raise ValueError(f"Unsupported precision: {precision}")

        shutil.copy2(working_path, final_model_path)

    ensure_onnx_opset(final_model_path, opset)
    optimize_onnx_basic(final_model_path, is_transformer)

    # infer_symbolic_shapes(final_model_path)
    infer_static_shapes(final_model_path)
    onnx.checker.check_model(str(final_model_path))

    if not keep_intermediate and raw_model_path != final_model_path:
        raw_model_path.unlink(missing_ok=True)

    return final_model_path


def get_onnx_input_name(model_path: Path) -> str:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for value_info in model.graph.input:
        if value_info.name not in initializer_names:
            return value_info.name
    raise RuntimeError(f"No runtime input found in {model_path}")


def list_calibration_images(directory: Path, limit: int) -> list[Path]:
    images = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No calibration images found in {directory}")
    return images[:limit]


class ImageCalibrationReader:
    """Small ONNX Runtime CalibrationDataReader-compatible image iterator."""

    def __init__(
        self,
        image_dir: Path,
        input_name: str,
        preprocess: Callable[[Path], np.ndarray],
        limit: int,
    ) -> None:
        self.input_name = input_name
        self.preprocess = preprocess
        self.image_paths = list_calibration_images(image_dir, limit)
        self._iterator: Iterator[Path] = iter(self.image_paths)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            image_path = next(self._iterator)
        except StopIteration:
            return None
        return {self.input_name: self.preprocess(image_path)}

    def rewind(self) -> None:
        self._iterator = iter(self.image_paths)


def imagenet_preprocessor(
    height: int,
    width: int,
    batch_size: int = 1,
) -> Callable[[Path], np.ndarray]:
    import torch
    from PIL import Image
    from torchvision.transforms import v2

    transform = v2.Compose(
        [
            v2.Resize((height, width), antialias=True),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def preprocess(path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        tensor = transform(image)
        if not hasattr(tensor, "unsqueeze"):
            raise TypeError("The torchvision transform did not return a tensor")
        array = tensor.unsqueeze(0).cpu().numpy().astype(np.float32)
        return np.repeat(array, batch_size, axis=0)

    return preprocess


def yolo_preprocessor(
    height: int,
    width: int,
    batch_size: int = 1,
) -> Callable[[Path], np.ndarray]:
    from PIL import Image

    def preprocess(path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        image.thumbnail((width, height))
        canvas = Image.new("RGB", (width, height), (114, 114, 114))
        offset = ((width - image.width) // 2, (height - image.height) // 2)
        canvas.paste(image, offset)
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        sample = np.transpose(array, (2, 0, 1))[None, ...]
        return np.repeat(sample, batch_size, axis=0)

    return preprocess


def hf_image_preprocessor(
    processor: Any,
    height: int,
    width: int,
) -> Callable[[Path], np.ndarray]:
    from PIL import Image

    def preprocess(path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB").resize((width, height))
        values = processor(images=image, return_tensors="np")["pixel_values"]
        return np.asarray(values, dtype=np.float32)

    return preprocess
