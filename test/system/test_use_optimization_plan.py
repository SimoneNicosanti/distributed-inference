"""Run every flow against the current published plan and a local reference."""

import os

import httpx
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from test.system.optimization_scenario import OptimizationScenario

from integration.dto.tensor.tensor_bundle_dto import TensorBundleDto
from integration.front_end.http.front_end_http_paths import FRONT_END_RUN_INFERENCE_PATH
from integration.front_end.http.front_end_http_schema import (
    RunInferenceRequest,
    RunInferenceResponse,
)


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


@pytest.mark.integration
def test_use_optimization_plan(optimization_scenario: OptimizationScenario) -> None:
    scenario = optimization_scenario
    model = onnx.load(scenario.model_path, load_external_data=False)
    bundle = _build_inference_bundle(model)
    inputs = bundle.to_ndarrays()
    reference = ort.InferenceSession(
        str(scenario.model_path), providers=["CPUExecutionProvider"]
    )
    expected = dict(
        zip(
            (output.name for output in reference.get_outputs()),
            reference.run(None, inputs),
            strict=True,
        )
    )
    rtol = float(os.getenv("INFERENCE_RTOL", "1e-3"))
    atol = float(os.getenv("INFERENCE_ATOL", "1e-5"))

    with httpx.Client(base_url=scenario.front_end_base_url, timeout=300.0) as client:
        for flow_id, plan in scenario.flow_plans.items():
            assert {spec.name for spec in plan.model_specs.inputs} == set(inputs)
            for spec in plan.model_specs.inputs:
                tensor = inputs[spec.name]
                assert np.dtype(spec.dtype) == tensor.dtype
                assert len(spec.shape) == tensor.ndim
                for declared, actual in zip(spec.shape, tensor.shape, strict=True):
                    if isinstance(declared, int):
                        assert declared == actual

            request = RunInferenceRequest(flow_id=flow_id, bundle=bundle)
            response = client.post(
                FRONT_END_RUN_INFERENCE_PATH, json=request.model_dump(mode="json")
            )
            assert response.status_code == 200, (
                f"Flow {flow_id}, plan {plan.plan_version}: "
                f"{response.status_code} {response.text}"
            )
            result = RunInferenceResponse.model_validate(response.json())
            outputs = result.bundle.to_ndarrays()
            assert set(outputs) == set(expected)
            for name, expected_tensor in expected.items():
                actual = outputs[name]
                assert actual.shape == expected_tensor.shape
                assert actual.dtype == expected_tensor.dtype
                assert np.isfinite(actual).all(), name
                if actual.dtype.kind in {"f", "c"}:
                    np.testing.assert_allclose(
                        actual, expected_tensor, rtol=rtol, atol=atol, err_msg=name
                    )
                else:
                    np.testing.assert_array_equal(actual, expected_tensor, err_msg=name)
