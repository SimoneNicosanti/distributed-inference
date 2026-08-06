import asyncio
import gc
from collections.abc import Sequence
from typing import override

import numpy as np
import onnxruntime as ort

from distributed_inference.worker.adapters.outbound.execution.onnx.onnx_sub_model_executor_options import (
    OnnxDeviceType,
    OnnxSubModelExecutorOptions,
)
from distributed_inference.worker.application.ports.outbound.sub_model_executor import (
    SubModelExecutor,
)
from distributed_inference.worker.domain.sub_model.execution.sub_model_execution_input_output import (
    SubModelExecutionInput,
    SubModelExecutionOutput,
)
from distributed_inference.worker.domain.tensor.tensor import Tensor, TensorBundle


class OnnxSubModelExecutor(SubModelExecutor):
    def __init__(
        self,
        inference_session: ort.InferenceSession,
        onnx_sub_model_executor_options: OnnxSubModelExecutorOptions,
    ) -> None:
        ## Build session to be reused across multiple requests to the same sub-model
        self._inference_session: ort.InferenceSession = inference_session

        self._inputs_meta_dict: dict[str, ort.NodeArg] = {
            item.name: item for item in inference_session.get_inputs()
        }

        self._output_names: list[str] = [
            item.name for item in inference_session.get_outputs()
        ]

        self._onnx_sub_model_executor_options = onnx_sub_model_executor_options

    @override
    async def close(self) -> None:
        del self._inference_session
        gc.collect()

    @override
    async def process_sub_model_inference_input(
        self, sub_model_execution_input: SubModelExecutionInput
    ) -> SubModelExecutionOutput:

        ## Three ways to run inference
        ## Using a thread in the same process
        ## Using a separate process
        ## Using a separate process in a process pool

        ## await asyncio.to_thread()
        ## await asyncio.create_subprocess_exec
        ## loop = asyncio.get_running_loop() + await loop.run_in_executor

        ## We just need a way to call await, otherwise we would block the main loop
        ## For now we use to thread, then maybe we can evaluate other options

        sub_model_execution_output = await asyncio.to_thread(
            self._process_sub_model_inference_input_sync, sub_model_execution_input
        )

        return sub_model_execution_output

    def _process_sub_model_inference_input_sync(
        self, sub_model_execution_input: SubModelExecutionInput
    ) -> SubModelExecutionOutput:

        ## TODO: With this management we are managing only numerical types input
        session_inputs: dict[str, np.ndarray] = {}

        for name, metadata in self._inputs_meta_dict.items():
            tensor = sub_model_execution_input.payload.get_tensor_by_name(name)

            if tensor is None:
                if metadata.type.startswith("optional("):
                    continue

                raise ValueError(f"Required input is missing: {name}")

            session_inputs[name] = tensor.get_value()

        device_type = self._map_device_type(
            self._onnx_sub_model_executor_options.device_type
        )
        device_id = self._onnx_sub_model_executor_options.device_id
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
                output_name: Tensor(
                    name=output_name,
                    shape=session_output[output_name].shape,
                    dtype=session_output[output_name].dtype,
                    value=session_output[output_name],
                )
                for output_name in self._output_names
            },
        )

        sub_model_execution_output = SubModelExecutionOutput(
            sub_model_execution_context=sub_model_execution_input.sub_model_execution_context,
            payload=tensor_bundle,
        )

        return sub_model_execution_output

    def _map_device_type(self, device_type: OnnxDeviceType) -> str:
        if device_type == OnnxDeviceType.CUDA:
            return "cuda"

        elif device_type == OnnxDeviceType.CPU:
            return "cpu"

        else:
            raise ValueError(f"Unknown device type: {device_type}")
