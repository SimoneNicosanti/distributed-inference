from typing import override

import onnxruntime as ort

from distributed_inference.worker.application.ports.outbound.sub_model_executor import (
    SubModelExecutor,
)
from distributed_inference.worker.application.ports.outbound.sub_model_executor_factory import (
    SubModelExecutorDeploymentOptions,
    SubModelExecutorFactory,
)


class OnnxSubModelExecutorFactory(SubModelExecutorFactory):
    @override
    async def create_sub_model_executor(
        self,
        deployment: SubModelExecutorDeploymentOptions,
    ) -> SubModelExecutor:

        ## Call inference session builder based on path and other options
        ## Create the inference worker
        ## Return it

        raise NotImplementedError


class OnnxInferenceSessionBuilder:
    @classmethod
    async def build_inference_session(cls) -> ort.InferenceSession:
        raise NotImplementedError
