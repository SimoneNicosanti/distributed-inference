import asyncio
import tempfile
from pathlib import Path
from typing import Any, Protocol, override

import numpy as np
import onnxruntime as ort

from artifacts.contracts.artifact_bundle import ArtifactBundle
from worker.adapters.outbound.profiling.model_execution.ort.parser.cpu_profile_parser import (
    parse_cpu_profile,
)
from worker.adapters.outbound.profiling.model_execution.ort.parser.gpu_profile_parser import (
    parse_gpu_profile,
)
from worker.adapters.outbound.profiling.model_execution.ort.parser.parsing_utils import (
    read_profile_events,
)
from worker.application.ports.outbound.profiling.model_execution.engine_model_execution_profiler import (
    EngineModelExecutionProfiler,
    ModelExecutionProfilingError,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    Backend,
    BackendExecutionProfile,
)
from worker.domain.profiling.model_execution.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)

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

PROFILE_FILE_PREFIX = "onnxruntime_profile"


## NOTE: Very important note about model profiling.
## It looks like ort uses a non monotonic clock for profiling in Linux and this can cause invalid layers execution durations.
## To cope with this we implemented the following logic:
## 1. We do not validate duration for warmup runs
## 2. If a layer of a profile iteration has negative duration, the whole run is not considered for averaging time
## 3. The average is done only on the valid runs
## Consequences:
## 1. We may average on less than PROFILE_ITERATIONS runs
## 2. If the clock drift does not cause negative durations, we might use lower times
## To handle the second, we might think about considering a different statistic (e.g., max or 99th percentile)


class OrtInputInfo(Protocol):
    name: str
    type: str


class OrtInputInfoProvider(Protocol):
    def get_inputs(self) -> list[OrtInputInfo]: ...


class OrtModelExecutionProfiler(EngineModelExecutionProfiler):
    def __init__(self) -> None:
        pass

    @override
    async def profile_model_execution_on_backend_with_input_shapes(
        self,
        model_static_profile: ModelStaticProfile,
        artifact_bundle: ArtifactBundle,
        backend: Backend,
        input_shapes: ModelInputConcreteShapes,
    ) -> BackendExecutionProfile:

        try:
            return await asyncio.to_thread(
                self._sync_profile_model_execution_on_backend_with_input_shapes,
                model_static_profile,
                artifact_bundle,
                backend,
                input_shapes,
            )
        except Exception as e:
            raise ModelExecutionProfilingError(
                f"Unable to profile model {model_static_profile.model_variant_id} "
                f"on backend {backend.value} "
                f"for shape {input_shapes.input_shapes}"
            ) from e

    def _sync_profile_model_execution_on_backend_with_input_shapes(
        self,
        model_static_profile: ModelStaticProfile,
        artifact_bundle: ArtifactBundle,
        backend: Backend,
        input_shapes: ModelInputConcreteShapes,
    ) -> BackendExecutionProfile:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            session = self._build_session_on_backend(
                artifact_bundle, backend, temp_dir_name
            )
            inputs = self._create_dummy_inputs_for_profiling(
                session, backend, input_shapes
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
        input_shapes: ModelInputConcreteShapes,
    ) -> dict[str, ort.OrtValue]:
        inputs: dict[str, ort.OrtValue] = {}
        for input_info in session.get_inputs():
            shape = input_shapes.input_shapes.get(input_info.name)
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
        profile_events = read_profile_events(profile_path)
        match backend:
            case Backend.CPU:
                return parse_cpu_profile(
                    profile_events,
                    warmup_iterations=WARMUP_ITERATIONS,
                    profile_iterations=PROFILE_ITERATIONS,
                )
            case Backend.GPU:
                return parse_gpu_profile(
                    profile_events,
                    warmup_iterations=WARMUP_ITERATIONS,
                    profile_iterations=PROFILE_ITERATIONS,
                )
            case _:
                raise ValueError(f"Unsupported backend: {backend!r}")
