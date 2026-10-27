import json
from pathlib import Path

import pytest

from worker.adapters.outbound.profiling.model_execution.ort_engine_model_execution_profiler import (
    PROFILE_ITERATIONS,
    OrtModelExecutionProfiler,
)
from worker.domain.profiling.model_execution.model_execution_profile import Backend


def _write_profile(path: Path, events: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps(events))
    return path


def test_parse_cpu_profile_uses_node_duration_and_discards_warmup(
    tmp_path: Path,
) -> None:
    events = [
        {
            "cat": "Node",
            "name": "layer_kernel_time",
            "dur": duration,
            "args": {
                "provider": "CPUExecutionProvider",
                "mem_bytes_in_use": str(memory_bytes),
            },
        }
        for duration, memory_bytes in zip(
            [1_000_000] * 10 + [2_000_000] * PROFILE_ITERATIONS,
            [8192] * 10 + [2048] * (PROFILE_ITERATIONS - 1) + [4096],
            strict=True,
        )
    ]
    events.append(
        {
            "cat": "Node",
            "name": "ignored_kernel_time",
            "dur": 9_000_000,
            "args": {
                "provider": "CUDAExecutionProvider",
                "mem_bytes_in_use": "8192",
            },
        }
    )

    profile = OrtModelExecutionProfiler._parse_profile_events_for_backend(
        _write_profile(tmp_path / "cpu.json", events), Backend.CPU
    )

    assert profile.backend is Backend.CPU
    assert profile.layer_profiles["layer"].execution_time_s == pytest.approx(2.0)
    assert profile.layer_profiles["layer"].memory_bytes == 4096
    assert "ignored" not in profile.layer_profiles


def test_parse_gpu_profile_sums_kernels_for_each_node_invocation(
    tmp_path: Path,
) -> None:
    events: list[dict[str, object]] = []
    for kernel_durations in [[1_000, 2_000]] * 10 + [[100, 200]] * PROFILE_ITERATIONS:
        events.extend(
            [
                {
                    "cat": "Node",
                    "name": "layer_kernel_time",
                    "dur": 50_000,
                    "args": {
                        "provider": "CUDAExecutionProvider",
                        "mem_bytes_in_use": "4096",
                    },
                },
                *[
                    {"cat": "Kernel", "name": "cuda_kernel", "dur": duration}
                    for duration in kernel_durations
                ],
            ]
        )

    profile = OrtModelExecutionProfiler._parse_profile_events_for_backend(
        _write_profile(tmp_path / "gpu.json", events), Backend.GPU
    )

    assert profile.backend is Backend.GPU
    assert profile.layer_profiles["layer"].execution_time_s == pytest.approx(0.0003)
    assert profile.layer_profiles["layer"].memory_bytes == 4096


def test_parse_gpu_profile_allows_cuda_nodes_without_kernels(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    for _ in range(10 + PROFILE_ITERATIONS):
        events.extend(
            [
                {
                    "cat": "Node",
                    "name": "Flatten_kernel_time",
                    "dur": 50,
                    "args": {
                        "provider": "CUDAExecutionProvider",
                        "mem_bytes_in_use": "0",
                    },
                },
                {
                    "cat": "Node",
                    "name": "cuda_layer_kernel_time",
                    "dur": 50_000,
                    "args": {
                        "provider": "CUDAExecutionProvider",
                        "mem_bytes_in_use": "4096",
                    },
                },
                {"cat": "Kernel", "name": "cuda_kernel", "dur": 300},
            ]
        )

    profile = OrtModelExecutionProfiler._parse_profile_events_for_backend(
        _write_profile(tmp_path / "gpu-zero-kernel-node.json", events), Backend.GPU
    )

    assert profile.layer_profiles["Flatten"].execution_time_s == 0.0
    assert profile.layer_profiles["Flatten"].memory_bytes == 0
    assert profile.layer_profiles["cuda_layer"].execution_time_s == pytest.approx(
        0.0003
    )


def test_parse_gpu_profile_includes_cpu_fallback_nodes(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    for _ in range(10 + PROFILE_ITERATIONS):
        events.extend(
            [
                {
                    "cat": "Node",
                    "name": "cuda_layer_kernel_time",
                    "dur": 50_000,
                    "args": {
                        "provider": "CUDAExecutionProvider",
                        "mem_bytes_in_use": "4096",
                    },
                },
                {"cat": "Kernel", "name": "cuda_kernel", "dur": 300},
                {
                    "cat": "Node",
                    "name": "cpu_layer_kernel_time",
                    "dur": 2_000,
                    "args": {
                        "provider": "CPUExecutionProvider",
                        "mem_bytes_in_use": "16384",
                    },
                },
            ]
        )

    profile = OrtModelExecutionProfiler._parse_profile_events_for_backend(
        _write_profile(tmp_path / "hybrid-gpu.json", events), Backend.GPU
    )

    assert profile.layer_profiles["cuda_layer"].execution_time_s == pytest.approx(
        0.0003
    )
    assert profile.layer_profiles["cuda_layer"].memory_bytes == 4096
    assert profile.layer_profiles["cpu_layer"].execution_time_s == pytest.approx(0.002)
    assert profile.layer_profiles["cpu_layer"].memory_bytes == 16384


def test_parse_profile_uses_none_when_allocator_memory_is_unavailable(
    tmp_path: Path,
) -> None:
    events = [
        {
            "cat": "Node",
            "name": "layer_kernel_time",
            "dur": 1_000,
            "args": {"provider": "CPUExecutionProvider"},
        }
        for _ in range(10 + PROFILE_ITERATIONS)
    ]

    profile = OrtModelExecutionProfiler._parse_profile_events_for_backend(
        _write_profile(tmp_path / "cpu-without-memory.json", events), Backend.CPU
    )

    assert profile.layer_profiles["layer"].memory_bytes is None


def test_parse_gpu_profile_rejects_cpu_only_execution(tmp_path: Path) -> None:
    events = [
        {
            "cat": "Node",
            "name": "cpu_layer_kernel_time",
            "dur": 2_000,
            "args": {"provider": "CPUExecutionProvider"},
        }
        for _ in range(10 + PROFILE_ITERATIONS)
    ]

    with pytest.raises(ValueError, match="no CUDA node events"):
        OrtModelExecutionProfiler._parse_profile_events_for_backend(
            _write_profile(tmp_path / "cpu-only-gpu.json", events), Backend.GPU
        )


def test_parse_gpu_profile_rejects_missing_device_events(tmp_path: Path) -> None:
    events = [
        {
            "cat": "Node",
            "name": "layer_kernel_time",
            "dur": 50_000,
            "args": {"provider": "CUDAExecutionProvider"},
        }
    ]

    with pytest.raises(ValueError, match="--enable_cuda_profiling"):
        OrtModelExecutionProfiler._parse_profile_events_for_backend(
            _write_profile(tmp_path / "gpu-without-kernels.json", events),
            Backend.GPU,
        )


@pytest.mark.parametrize("sample_count", [PROFILE_ITERATIONS, 36])
def test_parse_profile_rejects_unexpected_sample_count(
    tmp_path: Path,
    sample_count: int,
) -> None:
    events = [
        {
            "cat": "Node",
            "name": "layer_kernel_time",
            "dur": 1_000,
            "args": {"provider": "CPUExecutionProvider"},
        }
        for _ in range(sample_count)
    ]

    with pytest.raises(ValueError, match=f"Expected 35 .* got {sample_count}"):
        OrtModelExecutionProfiler._parse_profile_events_for_backend(
            _write_profile(tmp_path / f"cpu-{sample_count}.json", events),
            Backend.CPU,
        )
