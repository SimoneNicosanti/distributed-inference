import asyncio
from typing import override

import onnxruntime as ort

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.domain.plan import ResourceAllocation
from distributed_inference.model_optimizer.adapters.outbound.onnx_model_optimizer import (
    OnnxModelOptimizer,
)
from distributed_inference.worker.adapters.outbound.execution.onnx.onnx_sub_model_executor import (
    OnnxSubModelExecutor,
)
from distributed_inference.worker.adapters.outbound.execution.onnx.onnx_sub_model_executor_options import (
    OnnxDeviceType,
    OnnxSubModelExecutorOptions,
)
from distributed_inference.worker.application.ports.outbound.sub_model_executor import (
    SubModelExecutor,
)
from distributed_inference.worker.application.ports.outbound.sub_model_executor_factory import (
    SubModelExecutorFactory,
)


class OnnxInferenceSessionBuilder:
    @classmethod
    def build_inference_session(
        cls,
        materialized_artifact: MaterializedArtifact,
        resource_allocation: ResourceAllocation,
    ) -> ort.InferenceSession:
        ## TODO: We need should check for the different deployment options
        ## TODO: Here we should apply all the optimizations needed
        ## We can call the optimizer we have built for the graph extraction pipeline
        if resource_allocation.use_gpu:
            sess = ort.InferenceSession(
                materialized_artifact.entrypoint_path.as_posix(),
                providers=["CUDAExecutionProvider"],
            )
        else:
            sess = ort.InferenceSession(
                materialized_artifact.entrypoint_path.as_posix()
            )
        return sess


class OnnxSubModelExecutorFactory(SubModelExecutorFactory):
    def __init__(self, onnx_optimizer: OnnxModelOptimizer) -> None:
        self._onnx_optimizer = onnx_optimizer

    @override
    async def create_sub_model_executor(
        self,
        materialized_artifact: MaterializedArtifact,
        resource_allocation: ResourceAllocation,
    ) -> SubModelExecutor:

        ## Call inference session builder based on path and other options
        ## Create the inference worker
        ## Return it

        inference_session = await asyncio.to_thread(
            OnnxInferenceSessionBuilder.build_inference_session,
            materialized_artifact,
            resource_allocation,
        )

        onnx_sub_model_executor_options = OnnxSubModelExecutorOptions(
            device_type=OnnxDeviceType.CUDA
            if resource_allocation.use_gpu
            else OnnxDeviceType.CPU,
            device_id=0,
        )

        return OnnxSubModelExecutor(inference_session, onnx_sub_model_executor_options)
