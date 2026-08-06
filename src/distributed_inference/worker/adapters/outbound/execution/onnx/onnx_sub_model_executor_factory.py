import asyncio
from typing import override

import onnxruntime as ort

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.domain.plan import DeploymentOptions
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
        deployment_options: DeploymentOptions,
    ) -> ort.InferenceSession:
        ## TODO: We need should check for the different deployment options
        if deployment_options.use_gpu:
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
    @override
    async def create_sub_model_executor(
        self,
        materialized_artifact: MaterializedArtifact,
        deployment_options: DeploymentOptions,
    ) -> SubModelExecutor:

        ## Call inference session builder based on path and other options
        ## Create the inference worker
        ## Return it

        inference_session = await asyncio.to_thread(
            OnnxInferenceSessionBuilder.build_inference_session,
            materialized_artifact,
            deployment_options,
        )

        onnx_sub_model_executor_options = OnnxSubModelExecutorOptions(
            device_type=OnnxDeviceType.CUDA
            if deployment_options.use_gpu
            else OnnxDeviceType.CPU,
            device_id=0,
        )

        return OnnxSubModelExecutor(inference_session, onnx_sub_model_executor_options)
