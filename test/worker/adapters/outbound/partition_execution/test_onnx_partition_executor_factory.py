from pathlib import Path
from unittest.mock import Mock

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from artifacts.contracts.artifact_bundle import ArtifactBundle
from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from worker.adapters.outbound.partition_execution.onnx.onnx_partition_executor_factory import (
    OnnxInferenceSessionBuilder,
)
from worker.domain.plan.deployment_plan import ResourceAllocation


@pytest.mark.parametrize("use_gpu", (False, True))
def test_session_uses_the_allocated_provider_without_silent_gpu_fallback(
    monkeypatch: pytest.MonkeyPatch, use_gpu: bool
) -> None:
    provider = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
    bundle = Mock(spec=ArtifactBundle, entrypoint_path=Path("model.onnx"))
    session = Mock()
    session.get_providers.return_value = [provider, "CPUExecutionProvider"]
    create_session = Mock(return_value=session)
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )
    monkeypatch.setattr(ort, "InferenceSession", create_session)

    assert (
        OnnxInferenceSessionBuilder.build_inference_session(
            bundle, ResourceAllocation(use_gpu=use_gpu)
        )
        is session
    )
    create_session.assert_called_once_with(bundle.entrypoint_path, providers=[provider])
    assert session.disable_fallback.call_count == int(use_gpu)


def test_missing_cuda_provider_rejects_gpu_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    create_session = Mock()
    monkeypatch.setattr(ort, "InferenceSession", create_session)

    with pytest.raises(
        RuntimeError, match="provider unavailable: CUDAExecutionProvider"
    ):
        OnnxInferenceSessionBuilder.build_inference_session(
            Mock(spec=ArtifactBundle), ResourceAllocation(use_gpu=True)
        )
    create_session.assert_not_called()


def test_cuda_session_cannot_silently_initialize_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CUDAExecutionProvider"]
    )
    session = Mock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    monkeypatch.setattr(ort, "InferenceSession", Mock(return_value=session))

    with pytest.raises(
        RuntimeError, match="Expected primary provider 'CUDAExecutionProvider'"
    ):
        OnnxInferenceSessionBuilder.build_inference_session(
            Mock(spec=ArtifactBundle), ResourceAllocation(use_gpu=True)
        )


def test_cpu_allocation_runs_with_the_real_cpu_provider(tmp_path: Path) -> None:
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        "identity",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=10
    )
    path = tmp_path / "model.onnx"
    onnx.save_model(model, path)
    bundle = ArtifactBundle.from_workspace(
        ArtifactWorkspace(root_path=tmp_path, entrypoint_path=path)
    )

    session = OnnxInferenceSessionBuilder.build_inference_session(
        bundle, ResourceAllocation(use_gpu=False)
    )

    assert session.get_providers()[0] == "CPUExecutionProvider"
    value = np.array([2.0], dtype=np.float32)
    np.testing.assert_array_equal(session.run(None, {"input": value})[0], value)
