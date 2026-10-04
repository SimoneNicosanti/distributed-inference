import asyncio
import gc
from collections.abc import Sequence
from typing import override

import numpy as np
import onnxruntime as ort

from worker.adapters.outbound.partition_execution.onnx.onnx_partition_executor_options import (
    OnnxDeviceType,
    OnnxPartitionExecutorOptions,
)
from worker.application.ports.outbound.partition_executor import (
    PartitionExecutor,
)
from worker.domain.partition.partition_execution import (
    PartitionExecutionInput,
    PartitionExecutionOutput,
)
from worker.domain.partition.tensor_bundle import Tensor, TensorBundle


class OnnxPartitionExecutor(PartitionExecutor):
    def __init__(
        self,
        inference_session: ort.InferenceSession,
        options: OnnxPartitionExecutorOptions,
    ) -> None:
        ## Build session to be reused across multiple requests to the same partition.
        self._inference_session: ort.InferenceSession = inference_session

        self._inputs_meta_dict: dict[str, ort.NodeArg] = {
            item.name: item for item in inference_session.get_inputs()
        }

        self._output_names: list[str] = [
            item.name for item in inference_session.get_outputs()
        ]

        self._options = options

    @override
    async def close(self) -> None:
        del self._inference_session
        gc.collect()

    @override
    async def execute(
        self, partition_execution_input: PartitionExecutionInput
    ) -> PartitionExecutionOutput:

        ## Three ways to run inference
        ## Using a thread in the same process
        ## Using a separate process
        ## Using a separate process in a process pool

        ## await asyncio.to_thread()
        ## await asyncio.create_subprocess_exec
        ## loop = asyncio.get_running_loop() + await loop.run_in_executor

        ## We just need a way to call await, otherwise we would block the main loop
        ## For now we use to thread, then maybe we can evaluate other options

        partition_execution_output = await asyncio.to_thread(
            self._execute_sync, partition_execution_input
        )

        return partition_execution_output

    def _execute_sync(
        self, partition_execution_input: PartitionExecutionInput
    ) -> PartitionExecutionOutput:

        ## TODO: With this management we are managing only numerical types input
        session_inputs: dict[str, np.ndarray] = {}

        for name, metadata in self._inputs_meta_dict.items():
            tensor = partition_execution_input.payload.get_tensor_by_name(name)

            if tensor is None:
                if metadata.type.startswith("optional("):
                    continue

                raise ValueError(f"Required input is missing: {name}")

            session_inputs[name] = tensor.value

        device_type = self._map_device_type(self._options.device_type)
        device_id = self._options.device_id
        session_ort_inputs: dict[str, ort.OrtValue] = {
            name: ort.OrtValue.ortvalue_from_numpy(value, device_type, device_id)
            for name, value in session_inputs.items()
        }

        session_ort_outputs_list: Sequence[ort.OrtValue] = (
            self._inference_session.run_with_ort_values(
                self._output_names, session_ort_inputs
            )
        )

        session_output: dict[str, np.ndarray] = {
            output_name: session_ort_outputs_list[idx].numpy()
            for idx, output_name in enumerate(self._output_names)
        }

        tensor_bundle = TensorBundle(
            bundle={
                output_name: Tensor(value=session_output[output_name])
                for output_name in self._output_names
            },
        )

        partition_execution_output = PartitionExecutionOutput(
            partition_execution_context=partition_execution_input.partition_execution_context,
            payload=tensor_bundle,
        )

        return partition_execution_output

    def _map_device_type(self, device_type: OnnxDeviceType) -> str:
        if device_type == OnnxDeviceType.CUDA:
            return "cuda"

        elif device_type == OnnxDeviceType.CPU:
            return "cpu"

        else:
            raise ValueError(f"Unknown device type: {device_type}")
