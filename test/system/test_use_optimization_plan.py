"""Use PlanVersion(1) produced by test_optimization_request on a fresh stack."""

import os
import time
from pathlib import Path

import httpx
import numpy as np
import onnx
import pytest
from test.system.test_optimization_request import OPTIMIZATION_TEST_FLOW_IDS

from integration.dto.tensor.tensor_bundle_dto import TensorBundleDto
from integration.front_end.http.front_end_http_paths import (
    FRONT_END_RUN_INFERENCE_PATH,
)
from integration.front_end.http.front_end_http_schema import (
    RunInferenceRequest,
    RunInferenceResponse,
)

PLAN_VERSION = 1


def _build_inference_bundle(model: onnx.ModelProto) -> TensorBundleDto:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    tensors: dict[str, np.ndarray] = {}

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
        tensors[model_input.name] = np.zeros(shape, dtype=data_type)

    if not tensors:
        raise ValueError("ONNX model has no runtime inputs")

    return TensorBundleDto.from_ndarrays(tensors)


def _run_inference_when_plan_is_active(
    front_end_base_url: str,
    request: RunInferenceRequest,
    deployment_timeout_s: float,
) -> RunInferenceResponse:
    deadline = time.monotonic() + deployment_timeout_s

    with httpx.Client(base_url=front_end_base_url, timeout=300.0) as client:
        while True:
            response = client.post(
                FRONT_END_RUN_INFERENCE_PATH,
                json=request.model_dump(mode="json"),
            )
            if response.status_code == 200:
                return RunInferenceResponse.model_validate(response.json())

            if response.status_code < 500:
                pytest.fail(
                    f"Inference with plan {PLAN_VERSION} was rejected: "
                    f"{response.status_code} {response.text}"
                )

            if time.monotonic() >= deadline:
                pytest.fail(
                    f"Plan {PLAN_VERSION} did not become usable before timeout; "
                    f"last response: {response.status_code} {response.text}"
                )

            time.sleep(1.0)


@pytest.mark.integration
def test_use_optimization_plan_version_1() -> None:
    front_end_base_url = os.getenv("FRONT_END_BASE_URL")
    if front_end_base_url is None:
        pytest.skip("Set FRONT_END_BASE_URL to run the inference system test")

    model_path_value = os.getenv("MODEL_PATH")
    if model_path_value is None:
        pytest.skip("Set MODEL_PATH to run the inference system test")

    model_path = Path(model_path_value)
    if not model_path.is_file():
        pytest.fail(f"MODEL_PATH is not a file: {model_path}")

    model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(model)

    request = RunInferenceRequest(
        flow_id=OPTIMIZATION_TEST_FLOW_IDS[3],
        bundle=_build_inference_bundle(model),
    )
    response = _run_inference_when_plan_is_active(
        front_end_base_url,
        request,
        float(os.getenv("PLAN_DEPLOYMENT_TIMEOUT_S", "120")),
    )

    output_tensors = response.bundle.to_ndarrays()
    assert set(output_tensors) == {output.name for output in model.graph.output}
    assert all(tensor.size > 0 for tensor in output_tensors.values())
