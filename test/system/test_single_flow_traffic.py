"""Single-flow smoke test at the arrival rate submitted to the optimizer.

TRAFFIC_ARRIVAL_RATE defaults to 0.05 requests/s; TRAFFIC_REQUEST_COUNT to 10.
TRAFFIC_MAX_RESPONSE_TIME_S defaults to 100 seconds (mean response-time limit).
"""

import asyncio
import json
import math
import os
from pathlib import Path

import httpx
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from test.system.inference_support import _build_inference_bundle
from test.system.inference_traffic import TrafficSample, run_inference_traffic
from test.system.optimization_scenario import prepare_optimization_scenario

from integration.front_end.http.front_end_http_schema import (
    RunInferenceRequest,
    RunInferenceResponse,
)


@pytest.mark.integration
def test_single_flow_traffic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arrival_rate = float(os.getenv("TRAFFIC_ARRIVAL_RATE", "0.05"))
    request_count = int(os.getenv("TRAFFIC_REQUEST_COUNT", "10"))
    max_response_time_s = float(os.getenv("TRAFFIC_MAX_RESPONSE_TIME_S", "100"))
    assert math.isfinite(arrival_rate) and arrival_rate > 0
    assert request_count >= 2
    assert math.isfinite(max_response_time_s) and max_response_time_s > 0
    max_start_lag_s = float(
        os.getenv("TRAFFIC_MAX_START_LAG_S", str(max(0.1, 0.1 / arrival_rate)))
    )
    assert math.isfinite(max_start_lag_s) and max_start_lag_s > 0

    scenario = prepare_optimization_scenario(
        tmp_path,
        arrival_rates=(arrival_rate,),
        max_response_times_s=(max_response_time_s,),
    )
    assert len(scenario.flow_plans) == 1
    flow_id, plan = next(iter(scenario.flow_plans.items()))
    bundle = _build_inference_bundle(
        onnx.load(scenario.model_path, load_external_data=False)
    )
    reference = ort.InferenceSession(
        str(scenario.model_path), providers=["CPUExecutionProvider"]
    )
    expected = dict(
        zip(
            (output.name for output in reference.get_outputs()),
            reference.run(None, bundle.to_ndarrays()),
            strict=True,
        )
    )
    request = RunInferenceRequest(flow_id=flow_id, bundle=bundle)

    async def generate() -> tuple[TrafficSample, ...]:
        async with httpx.AsyncClient(
            base_url=scenario.front_end_base_url,
            timeout=300.0,
            limits=httpx.Limits(max_connections=request_count),
        ) as client:
            return await run_inference_traffic(
                client,
                request,
                arrival_rate=arrival_rate,
                request_count=request_count,
            )

    with capsys.disabled():
        print(
            f"Traffic: flow={flow_id}, arrival_rate={arrival_rate} requests/s, "
            f"requests={request_count}, send_window_s={(request_count - 1) / arrival_rate:.3f}",
            flush=True,
        )
    samples = asyncio.run(generate())
    rtol = float(os.getenv("INFERENCE_RTOL", "1e-3"))
    atol = float(os.getenv("INFERENCE_ATOL", "1e-5"))
    failures = []
    for sample in samples:
        try:
            assert sample.error is None, sample.error
            assert sample.response is not None
            assert sample.response.status_code == 200, (
                f"HTTP {sample.response.status_code}: {sample.response.text}"
            )
            outputs = RunInferenceResponse.model_validate(
                sample.response.json()
            ).bundle.to_ndarrays()
            assert set(outputs) == set(expected)
            for name, expected_tensor in expected.items():
                actual = outputs[name]
                assert actual.shape == expected_tensor.shape, name
                assert actual.dtype == expected_tensor.dtype, name
                assert np.isfinite(actual).all(), name
                if actual.dtype.kind in {"f", "c"}:
                    np.testing.assert_allclose(
                        actual, expected_tensor, rtol=rtol, atol=atol, err_msg=name
                    )
                else:
                    np.testing.assert_array_equal(actual, expected_tensor, err_msg=name)
        except (AssertionError, ValueError) as error:
            failures.append(f"Request {sample.request_number}: {error}")

    response_times = np.array([sample.response_time_s for sample in samples])
    max_start_lag = max(
        sample.started_at_s - sample.scheduled_at_s for sample in samples
    )
    summary = {
        "flow_id": str(flow_id),
        "plan_version": plan.plan_version.version_number,
        "target_arrival_rate": arrival_rate,
        "actual_arrival_rate": (request_count - 1)
        / (samples[-1].started_at_s - samples[0].started_at_s),
        "requests": len(samples),
        "valid_responses": len(samples) - len(failures),
        "max_start_lag_s": max_start_lag,
        "first_response_time_s": float(response_times[0]),
        "mean_response_time_s": float(response_times.mean()),
        "p95_response_time_s": float(np.percentile(response_times, 95)),
        "max_response_time_s": float(response_times.max()),
        "mean_response_time_limit_s": max_response_time_s,
    }
    with capsys.disabled():
        print(f"Traffic results: {json.dumps(summary, sort_keys=True)}", flush=True)
    assert not failures, "\n".join(failures)
    assert len(samples) == request_count
    assert max_start_lag <= max_start_lag_s, (
        f"Traffic generator missed its schedule: {max_start_lag:.3f}s > {max_start_lag_s:.3f}s"
    )
    assert float(response_times.mean()) <= max_response_time_s, summary
