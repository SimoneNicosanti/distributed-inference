"""Shared input generation for inference system tests."""

import numpy as np
import onnx

from integration.dto.tensor.tensor_bundle_dto import TensorBundleDto


def _build_inference_bundle(model: onnx.ModelProto) -> TensorBundleDto:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    tensors: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)

    for model_input in model.graph.input:
        if model_input.name in initializer_names:
            continue

        tensor_type = model_input.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
            raise ValueError(f"Input {model_input.name!r} has an undefined data type")

        data_type = np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        )
        if data_type.kind in {"b", "O", "S", "U", "V"}:
            raise ValueError(
                f"Input {model_input.name!r} uses unsupported dtype {data_type}"
            )

        shape = tuple(
            dimension.dim_value
            if dimension.HasField("dim_value") and dimension.dim_value > 0
            else 1
            for dimension in tensor_type.shape.dim
        )
        tensors[model_input.name] = (
            rng.uniform(0.0, 1.0, size=shape).astype(data_type)
            if data_type.kind == "f"
            else np.zeros(shape, dtype=data_type)
        )

    if not tensors:
        raise ValueError("ONNX model has no runtime inputs")

    return TensorBundleDto.from_ndarrays(tensors)
