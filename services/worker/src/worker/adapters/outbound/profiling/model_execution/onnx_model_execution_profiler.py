import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Protocol, override

import numpy as np
import onnxruntime as ort

from artifacts.contracts.artifact_bundle import ArtifactBundle
from worker.application.ports.outbound.profiling.model_execution.model_execution_profiler import (
    ModelExecutionProfiler,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    BackendLayerExecutionProfile,
    LayerExecutionProfile,
    ModelExecutionProfile,
    ShapeExecutionProfile,
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
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"


class OrtInputInfo(Protocol):
    name: str
    type: str


class OrtInputInfoProvider(Protocol):
    def get_inputs(self) -> list[OrtInputInfo]: ...


class OnnxModelExecutionProfiler(ModelExecutionProfiler):
    @override
    async def profile_model_execution(
        self, model_static_profile: ModelStaticProfile, artifact_bundle: ArtifactBundle
    ) -> ModelExecutionProfile:
        shape_execution_profiles = tuple(
            self._profile_shape_point_cpu(
                artifact_bundle.entrypoint_path,
                shape_point,
            )
            for shape_point in model_static_profile.shape_points
        )

        return ModelExecutionProfile(
            model_version_id=model_static_profile.model_variant_id,
            shape_execution_profiles=shape_execution_profiles,
        )

    @staticmethod
    def _profile_shape_point_cpu(
        model_path: Path,
        shape_point: ShapePoint,
    ) -> ShapeExecutionProfile:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            session_options = ort.SessionOptions()
            session_options.enable_profiling = True
            session_options.profile_file_prefix = str(
                Path(temp_dir_name).joinpath("onnxruntime_profile")
            )
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )

            session = ort.InferenceSession(
                model_path.as_posix(),
                sess_options=session_options,
                providers=[CPU_EXECUTION_PROVIDER],
            )
            inputs = OnnxModelExecutionProfiler._create_inputs(session, shape_point)

            for _ in range(WARMUP_ITERATIONS + PROFILE_ITERATIONS):
                session.run(None, inputs)

            profile_path = Path(session.end_profiling())
            profile_events = json.loads(profile_path.read_text())

        durations_by_layer: dict[str, list[float]] = defaultdict(list)
        for event in profile_events:
            event_name = event.get("name")
            event_arguments = event.get("args", {})
            if (
                event.get("cat") != NODE_EVENT_CATEGORY
                or not isinstance(event_name, str)
                or not event_name.endswith(NODE_EVENT_SUFFIX)
                or event_arguments.get("provider") != CPU_EXECUTION_PROVIDER
            ):
                continue

            layer_name = event_name.removesuffix(NODE_EVENT_SUFFIX)
            durations_by_layer[layer_name].append(float(event["dur"]) / 1_000_000)

        layer_profiles = {}
        for layer_name, durations in durations_by_layer.items():
            measured_durations = durations[-PROFILE_ITERATIONS:]
            if len(measured_durations) != PROFILE_ITERATIONS:
                raise ValueError(
                    f"Expected {PROFILE_ITERATIONS} profiling samples for "
                    f"layer {layer_name!r}, got {len(measured_durations)}"
                )

            layer_profiles[layer_name] = LayerExecutionProfile(
                layer_key=layer_name,
                cpu_execution_profile=BackendLayerExecutionProfile(
                    execution_time=float(np.mean(measured_durations)),
                    memory=0,
                ),
                gpu_execution_profile=None,
            )

        return ShapeExecutionProfile(
            shape_point=shape_point,
            layer_profiles=layer_profiles,
        )

    @staticmethod
    def _create_inputs(
        session: OrtInputInfoProvider,
        shape_point: ShapePoint,
    ) -> dict[str, np.ndarray]:
        inputs: dict[str, np.ndarray] = {}
        for input_info in session.get_inputs():
            shape = shape_point.input_shape_points.get(input_info.name)
            if shape is None:
                raise ValueError(
                    f"Missing concrete shape for model input {input_info.name!r}"
                )

            numpy_type = ORT_TO_NUMPY_TYPE.get(input_info.type)
            if numpy_type is None:
                raise ValueError(f"Unsupported ONNX input type: {input_info.type!r}")

            inputs[input_info.name] = np.ones(
                shape,
                dtype=numpy_type,
            )

        return inputs
