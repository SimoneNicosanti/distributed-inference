"""Run initial and repeated inference against published plans and a local reference."""

import os

import httpx
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from test.system.existing_plan_scenario import ExistingPlanScenario
from test.system.inference_support import _build_inference_bundle

from integration.front_end.http.front_end_http_paths import FRONT_END_RUN_INFERENCE_PATH
from integration.front_end.http.front_end_http_schema import (
    RunInferenceRequest,
    RunInferenceResponse,
)


@pytest.mark.integration
def test_use_optimization_plan(existing_plan_scenario: ExistingPlanScenario) -> None:
    scenario = existing_plan_scenario
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

            for attempt in (1, 2):
                request = RunInferenceRequest(flow_id=flow_id, bundle=bundle)
                response = client.post(
                    FRONT_END_RUN_INFERENCE_PATH, json=request.model_dump(mode="json")
                )
                assert response.status_code == 200, (
                    f"Flow {flow_id}, plan {plan.plan_version}, request {attempt}: "
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
                        np.testing.assert_array_equal(
                            actual, expected_tensor, err_msg=name
                        )
