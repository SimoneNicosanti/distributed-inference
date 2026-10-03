import gc
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import override

import aiofiles
import numpy as np
import onnx
import onnxruntime as ort

from artifacts.contracts.artifact_bundle import ArtifactBundle
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from worker.application.ports.outbound.profiling.model_execution.model_execution_profiler import (
    ModelExecutionProfiler,
)
from worker.domain.profiling.model_execution_profile import (
    BackendLayerExecutionProfile,
    LayerExecutionProfile,
    ModelExecutionProfile,
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


class OnnxModelExecutionProfiler(ModelExecutionProfiler):
    @override
    async def profile_model_execution(
        self, model_version_id: ModelVariantId, artifact_bundle: ArtifactBundle
    ) -> ModelExecutionProfile:

        entrypoint_path = artifact_bundle.entrypoint_path
        model = onnx.load_model(entrypoint_path, load_external_data=False)

        layer_profiles = {}
        async for (
            layer_name,
            single_layer_onnx_model,
        ) in self._get_single_layer_sub_models(model, entrypoint_path):
            layer_execution_profile = await self._profile_single_layer_execution(
                layer_name, single_layer_onnx_model
            )

            layer_profiles[layer_name] = layer_execution_profile

        return ModelExecutionProfile(
            model_version_id=model_version_id,
            layer_profiles=layer_profiles,
        )

    async def _get_single_layer_sub_models(
        self, model: onnx.ModelProto, model_path: Path
    ) -> AsyncGenerator[tuple[LayerKey, onnx.ModelProto]]:

        initializer_names = {
            initializer.name for initializer in model.graph.initializer
        }

        async with aiofiles.tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            for node in model.graph.node:
                input_names = set(node.input) - initializer_names
                output_names = set(node.output)

                tmp_model_path = tmp_dir.joinpath(f"{node.name}.onnx")

                onnx.utils.extract_model(
                    model_path, tmp_model_path, list(input_names), list(output_names)
                )

                node_onnx_model = onnx.load_model(tmp_model_path)
                yield node.name, node_onnx_model

    async def _profile_single_layer_execution(
        self, layer_name: str, node_onnx_model: onnx.ModelProto
    ) -> LayerExecutionProfile:

        ## TODO: Implement the GPU profiling logic

        cpu_profile = await self._profile_single_layer_execution_cpu(node_onnx_model)

        return LayerExecutionProfile(
            layer_key=layer_name,
            cpu_execution_profile=cpu_profile,
            gpu_execution_profile=None,
        )

    async def _profile_single_layer_execution_cpu(
        self, node_onnx_model: onnx.ModelProto
    ) -> BackendLayerExecutionProfile:

        ## TODO: Here we are assuming that there is no dynamic shape
        session = ort.InferenceSession(
            node_onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        input_dict: dict[str, ort.OrtValue] = {}
        for input_info in session.get_inputs():
            numpy_type = ORT_TO_NUMPY_TYPE[input_info.type]

            input_array = np.ones(
                input_info.shape,
                dtype=numpy_type,
            )

            input_dict[input_info.name] = ort.OrtValue.ortvalue_from_numpy(input_array)

        for _ in range(WARMUP_ITERATIONS):
            session.run_with_ort_values(None, input_dict)

        times = np.zeros(PROFILE_ITERATIONS)
        for i in range(PROFILE_ITERATIONS):
            start = time.perf_counter_ns()
            session.run_with_ort_values(None, input_dict)
            end = time.perf_counter_ns()
            times[i] = (end - start) / 1e9

        del session
        gc.collect()

        return BackendLayerExecutionProfile(
            execution_time=float(np.mean(times)), memory=0
        )
