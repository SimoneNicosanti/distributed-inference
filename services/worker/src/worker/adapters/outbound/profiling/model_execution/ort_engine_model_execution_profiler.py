import asyncio
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, cast, override

import numpy as np
import onnxruntime as ort

from artifacts.contracts.artifact_bundle import ArtifactBundle
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from worker.application.ports.outbound.profiling.model_execution.engine_model_execution_profiler import (
    EngineModelExecutionProfiler,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    Backend,
    BackendExecutionProfile,
    LayerExecutionProfile,
)
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)
from worker.domain.profiling.model_execution.shape_point import ShapePoint

ORT_TO_NUMPY_TYPE = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint64)": np.uint64,
    "tensor(uint32)": np.uint32,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}

WARMUP_ITERATIONS = 10

PROFILE_ITERATIONS = 25

NODE_EVENT_CATEGORY = "Node"
NODE_EVENT_SUFFIX = "_kernel_time"
KERNEL_EVENT_CATEGORY = "Kernel"
MEMORY_BYTES_IN_USE_ARGUMENT = "mem_bytes_in_use"

PROFILE_DURATION_MICROSECONDS_PER_SECOND = 1_000_000

PROFILE_FILE_PREFIX = "onnxruntime_profile"


class OrtInputInfo(Protocol):
    name: str
    type: str


class OrtInputInfoProvider(Protocol):
    def get_inputs(self) -> list[OrtInputInfo]: ...


type BackendSessionKey = tuple[Backend, ModelVariantId]


class OrtModelExecutionProfiler(EngineModelExecutionProfiler):
    def __init__(self) -> None:
        pass

    @override
    async def profile_model_execution_on_backend_with_shape_point(
        self,
        model_static_profile: ModelStaticProfile,
        artifact_bundle: ArtifactBundle,
        backend: Backend,
        shape_point: ShapePoint,
    ) -> BackendExecutionProfile:

        return await asyncio.to_thread(
            self._sync_profile_model_execution_on_backend_with_shape_point,
            model_static_profile,
            artifact_bundle,
            backend,
            shape_point,
        )

    def _sync_profile_model_execution_on_backend_with_shape_point(
        self,
        model_static_profile: ModelStaticProfile,
        artifact_bundle: ArtifactBundle,
        backend: Backend,
        shape_point: ShapePoint,
    ) -> BackendExecutionProfile:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            session = self._build_session_on_backend(
                artifact_bundle, backend, temp_dir_name
            )
            inputs = self._create_dummy_inputs_for_profiling(
                session, backend, shape_point
            )

            for _ in range(WARMUP_ITERATIONS + PROFILE_ITERATIONS):
                session.run_with_ort_values(None, inputs)

            self._dismiss_session_and_inputs(session, inputs)

            profile_path = Path(session.end_profiling())

            return self._parse_profile_events_for_backend(profile_path, backend)

    def _build_session_on_backend(
        self, artifact_bundle: ArtifactBundle, backend: Backend, temp_dir_name: str
    ) -> ort.InferenceSession:
        provider = self._match_backend_to_provider(backend)
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"ONNX Runtime provider unavailable: {provider}")

        session_options = ort.SessionOptions()
        session_options.enable_profiling = True
        session_options.profile_file_prefix = str(
            Path(temp_dir_name).joinpath(PROFILE_FILE_PREFIX)
        )

        ## TODO: Check what kind of optimizations we want to enable
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )

        model_path = artifact_bundle.entrypoint_path

        session = ort.InferenceSession(
            model_path.as_posix(),
            sess_options=session_options,
            providers=[provider],
        )

        active_providers = session.get_providers()
        if not active_providers or active_providers[0] != provider:
            raise RuntimeError(
                f"Expected primary provider {provider!r}, got {active_providers!r}"
            )

        if backend is Backend.GPU:
            session.disable_fallback()

        return session

    @staticmethod
    def _create_dummy_inputs_for_profiling(
        session: OrtInputInfoProvider,
        backend: Backend,
        shape_point: ShapePoint,
    ) -> dict[str, ort.OrtValue]:
        inputs: dict[str, ort.OrtValue] = {}
        for input_info in session.get_inputs():
            shape = shape_point.input_shape_points.get(input_info.name)
            if shape is None:
                raise ValueError(
                    f"Missing concrete shape for model input {input_info.name!r}"
                )

            numpy_type = ORT_TO_NUMPY_TYPE.get(input_info.type)
            if numpy_type is None:
                raise ValueError(f"Unsupported ONNX input type: {input_info.type!r}")

            numpy_input: np.ndarray[tuple[int, ...], np.dtype[Any]] = np.ones(
                shape, dtype=numpy_type
            )

            inputs[input_info.name] = ort.OrtValue.ortvalue_from_numpy(
                numpy_input,
                device_type=OrtModelExecutionProfiler._match_backend_to_device(backend),
            )

        return inputs

    def _dismiss_session_and_inputs(
        self, session: ort.InferenceSession, inputs: dict[str, ort.OrtValue]
    ) -> None:
        import gc

        del session
        for input in inputs.values():
            del input
        del inputs

        gc.collect()

    @staticmethod
    def _match_backend_to_device(backend: Backend) -> str:
        match backend:
            case Backend.CPU:
                return "cpu"
            case Backend.GPU:
                return "cuda"
            case _:
                raise ValueError(f"Unsupported backend: {backend!r}")

    @staticmethod
    def _match_backend_to_provider(backend: Backend) -> str:
        match backend:
            case Backend.CPU:
                return "CPUExecutionProvider"
            case Backend.GPU:
                return "CUDAExecutionProvider"
            case _:
                raise ValueError(f"Unsupported backend: {backend!r}")

    @staticmethod
    def _parse_profile_events_for_backend(
        profile_path: Path, backend: Backend
    ) -> BackendExecutionProfile:
        profile_events = OrtModelExecutionProfiler._read_profile_events(profile_path)

        durations_by_layer = OrtModelExecutionProfiler._parse_durations_for_backend(
            backend, profile_events
        )
        memory_bytes_by_layer = (
            OrtModelExecutionProfiler._parse_memory_bytes_in_use_by_layer(
                backend, profile_events
            )
        )

        expected_samples = WARMUP_ITERATIONS + PROFILE_ITERATIONS
        layer_profiles: dict[LayerKey, LayerExecutionProfile] = {}
        for layer_key, durations in durations_by_layer.items():
            if len(durations) != expected_samples:
                raise ValueError(
                    f"Expected {expected_samples} profiling samples for "
                    f"layer {layer_key!r}, got {len(durations)}"
                )

            memory_samples = memory_bytes_by_layer.get(layer_key, [])
            if len(memory_samples) != expected_samples:
                raise ValueError(
                    f"Expected {expected_samples} memory samples for "
                    f"layer {layer_key!r}, got {len(memory_samples)}"
                )

            measured_durations = durations[WARMUP_ITERATIONS:]
            measured_memory_samples = memory_samples[WARMUP_ITERATIONS:]
            available_memory_samples = [
                memory_bytes
                for memory_bytes in measured_memory_samples
                if memory_bytes is not None
            ]
            if available_memory_samples and len(available_memory_samples) != len(
                measured_memory_samples
            ):
                raise ValueError(f"Inconsistent memory samples for layer {layer_key!r}")

            layer_profiles[layer_key] = LayerExecutionProfile(
                execution_time_s=float(np.mean(measured_durations)),
                memory_bytes=(
                    max(available_memory_samples) if available_memory_samples else None
                ),
            )

        return BackendExecutionProfile(backend=backend, layer_profiles=layer_profiles)

    @staticmethod
    def _parse_memory_bytes_in_use_by_layer(
        backend: Backend, profile_events: list[dict[str, object]]
    ) -> dict[LayerKey, list[int | None]]:
        primary_provider = OrtModelExecutionProfiler._match_backend_to_provider(backend)
        providers = (
            {primary_provider, "CPUExecutionProvider"}
            if backend is Backend.GPU
            else {primary_provider}
        )
        memory_bytes_by_layer: dict[LayerKey, list[int | None]] = defaultdict(list)

        for event in profile_events:
            event_arguments = event.get("args")
            if not isinstance(event_arguments, dict):
                continue

            provider = event_arguments.get("provider")
            if not isinstance(provider, str) or provider not in providers:
                continue

            layer_key = OrtModelExecutionProfiler._node_layer_key_for_provider(
                event, provider
            )
            if layer_key is not None:
                memory_bytes_by_layer[layer_key].append(
                    OrtModelExecutionProfiler._event_memory_bytes_in_use(event)
                )

        return dict(memory_bytes_by_layer)

    @staticmethod
    def _event_memory_bytes_in_use(event: dict[str, object]) -> int | None:
        event_arguments = cast(dict[object, object], event["args"])
        raw_memory_bytes = event_arguments.get(MEMORY_BYTES_IN_USE_ARGUMENT)
        if raw_memory_bytes is None:
            return None

        if isinstance(raw_memory_bytes, bool) or not isinstance(
            raw_memory_bytes, (int, str)
        ):
            raise ValueError("ONNX Runtime profile has invalid memory bytes in use")

        try:
            memory_bytes = int(raw_memory_bytes)
        except ValueError as error:
            raise ValueError(
                "ONNX Runtime profile has invalid memory bytes in use"
            ) from error

        if memory_bytes < 0:
            raise ValueError("ONNX Runtime profile has negative memory bytes in use")

        return memory_bytes

    @staticmethod
    def _parse_durations_for_backend(
        backend: Backend, profile_events: list[dict[str, object]]
    ) -> dict[LayerKey, list[float]]:
        match backend:
            case Backend.CPU:
                durations_by_layer = (
                    OrtModelExecutionProfiler._parse_cpu_durations_by_layer(
                        profile_events
                    )
                )
            case Backend.GPU:
                durations_by_layer = (
                    OrtModelExecutionProfiler._parse_gpu_durations_by_layer(
                        profile_events
                    )
                )
            case _:
                raise ValueError(f"Unsupported backend: {backend!r}")
        return durations_by_layer

    @staticmethod
    def _read_profile_events(profile_path: Path) -> list[dict[str, object]]:
        parsed_profile = cast(object, json.loads(profile_path.read_text()))
        if not isinstance(parsed_profile, list):
            raise ValueError("ONNX Runtime profile must contain a JSON array")
        if not all(isinstance(event, dict) for event in parsed_profile):
            raise ValueError("ONNX Runtime profile contains a non-object event")

        return cast(list[dict[str, object]], parsed_profile)

    @staticmethod
    def _parse_cpu_durations_by_layer(
        profile_events: list[dict[str, object]],
    ) -> dict[LayerKey, list[float]]:
        durations_by_layer: dict[LayerKey, list[float]] = defaultdict(list)
        for event in profile_events:
            layer_key = OrtModelExecutionProfiler._node_layer_key_for_provider(
                event, "CPUExecutionProvider"
            )
            if layer_key is not None:
                durations_by_layer[layer_key].append(
                    OrtModelExecutionProfiler._event_duration_seconds(event)
                )

        return dict(durations_by_layer)

    @staticmethod
    def _parse_gpu_durations_by_layer(
        profile_events: list[dict[str, object]],
    ) -> dict[LayerKey, list[float]]:
        durations_by_layer: dict[LayerKey, list[float]] = defaultdict(list)
        current_layer_key: LayerKey | None = None
        current_kernel_durations: list[float] = []
        found_cuda_node = False
        found_kernel_event = False

        def append_current_sample() -> None:
            if current_layer_key is None:
                return
            durations_by_layer[current_layer_key].append(
                sum(current_kernel_durations, 0.0)
            )

        for event in profile_events:
            if event.get("cat") == NODE_EVENT_CATEGORY:
                append_current_sample()
                current_layer_key = (
                    OrtModelExecutionProfiler._node_layer_key_for_provider(
                        event, "CUDAExecutionProvider"
                    )
                )
                current_kernel_durations = []
                if current_layer_key is not None:
                    found_cuda_node = True
                    continue

                cpu_layer_key = OrtModelExecutionProfiler._node_layer_key_for_provider(
                    event, "CPUExecutionProvider"
                )
                if cpu_layer_key is not None:
                    durations_by_layer[cpu_layer_key].append(
                        OrtModelExecutionProfiler._event_duration_seconds(event)
                    )
            elif (
                current_layer_key is not None
                and event.get("cat") == KERNEL_EVENT_CATEGORY
            ):
                found_kernel_event = True
                current_kernel_durations.append(
                    OrtModelExecutionProfiler._event_duration_seconds(event)
                )

        append_current_sample()
        if not found_cuda_node:
            raise ValueError("GPU profile contains no CUDA node events")
        if not found_kernel_event:
            raise ValueError(
                "GPU profile contains no kernel events; ONNX Runtime must be built "
                "with --enable_cuda_profiling and CUPTI must be available"
            )

        return dict(durations_by_layer)

    @staticmethod
    def _node_layer_key_for_provider(
        event: dict[str, object], provider: str
    ) -> LayerKey | None:
        event_name = event.get("name")
        event_arguments = event.get("args")
        if (
            event.get("cat") != NODE_EVENT_CATEGORY
            or not isinstance(event_name, str)
            or not event_name.endswith(NODE_EVENT_SUFFIX)
            or not isinstance(event_arguments, dict)
            or event_arguments.get("provider") != provider
        ):
            return None

        return event_name.removesuffix(NODE_EVENT_SUFFIX)

    @staticmethod
    def _event_duration_seconds(event: dict[str, object]) -> float:
        duration = event.get("dur")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError("ONNX Runtime profile event has an invalid duration")
        if duration < 0:
            raise ValueError("ONNX Runtime profile event has a negative duration")

        return float(duration) / PROFILE_DURATION_MICROSECONDS_PER_SECOND
