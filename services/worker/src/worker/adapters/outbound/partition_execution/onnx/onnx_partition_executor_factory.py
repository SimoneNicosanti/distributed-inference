import asyncio
from typing import override

import onnxruntime as ort

from artifacts.contracts.artifact_bundle import ArtifactBundle
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
    PartitionExecutorCreationError,
    PartitionExecutorFactory,
)
from worker.domain.plan.deployment_plan import ResourceAllocation


class OnnxInferenceSessionBuilder:
    @classmethod
    def build_inference_session(
        cls,
        bundle: ArtifactBundle,
        resource_allocation: ResourceAllocation,
    ) -> ort.InferenceSession:
        ## TODO: We should check for the different deployment options
        ## TODO: Here we should apply all the optimizations needed

        provider = (
            "CUDAExecutionProvider"
            if resource_allocation.use_gpu
            else "CPUExecutionProvider"
        )
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"ONNX Runtime provider unavailable: {provider}")
        sess = ort.InferenceSession(bundle.entrypoint_path, providers=[provider])
        active_providers = sess.get_providers()
        if not active_providers or active_providers[0] != provider:
            raise RuntimeError(
                f"Expected primary provider {provider!r}, got {active_providers!r}"
            )
        if resource_allocation.use_gpu:
            sess.disable_fallback()
        return sess


class OnnxPartitionExecutorFactory(PartitionExecutorFactory):
    @override
    async def create(
        self,
        bundle: ArtifactBundle,
        resource_allocation: ResourceAllocation,
    ) -> PartitionExecutor:
        try:
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
        except Exception as e:
            raise PartitionExecutorCreationError(
                "Unable to create ONNX partition executor"
            ) from e
