import asyncio
from typing import override

import onnxruntime as ort

from artifacts.contracts.artifact_bundle import ArtifactBundle
from shared.plan.plan import ResourceAllocation
from worker.adapters.outbound.partition_execution.onnx.onnx_partition_executor import (
    OnnxPartitionExecutor,
)
from worker.adapters.outbound.partition_execution.onnx.onnx_partition_executor_options import (
    OnnxDeviceType,
    OnnxPartitionExecutorOptions,
)
from worker.application.ports.outbound.partition_execution.partition_executor import (
    PartitionExecutor,
)
from worker.application.ports.outbound.partition_execution.partition_executor_factory import (
    PartitionExecutorFactory,
)


class OnnxInferenceSessionBuilder:
    @classmethod
    def build_inference_session(
        cls,
        bundle: ArtifactBundle,
        resource_allocation: ResourceAllocation,
    ) -> ort.InferenceSession:
        ## TODO: We need should check for the different deployment options
        ## TODO: Here we should apply all the optimizations needed
        ## We can call the optimizer we have built for the graph extraction pipeline

        if resource_allocation.use_gpu:
            sess = ort.InferenceSession(
                bundle.entrypoint_path,
                providers=["CUDAExecutionProvider"],
            )
        else:
            sess = ort.InferenceSession(bundle.entrypoint_path)
        return sess


class OnnxPartitionExecutorFactory(PartitionExecutorFactory):
    @override
    async def create(
        self,
        bundle: ArtifactBundle,
        resource_allocation: ResourceAllocation,
    ) -> PartitionExecutor:

        ## Call inference session builder based on path and other options
        ## Create the inference worker
        ## Return it

        inference_session = await asyncio.to_thread(
            OnnxInferenceSessionBuilder.build_inference_session,
            bundle,
            resource_allocation,
        )

        options = OnnxPartitionExecutorOptions(
            device_type=OnnxDeviceType.CUDA
            if resource_allocation.use_gpu
            else OnnxDeviceType.CPU,
            device_id=0,
        )

        return OnnxPartitionExecutor(inference_session, options)
