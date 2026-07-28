from typing import override

import onnxruntime as ort

from distributed_inference.application.worker.contracts.execution.inference.inference_worker import (
    InferenceWorker,
)
from distributed_inference.application.worker.contracts.execution.inference.inference_worker_factory import (
    InferenceWorkerDeploymentOptions,
    InferenceWorkerFactory,
)


class OnnxInferenceWorkerFactory(InferenceWorkerFactory):
    @override
    async def create_inference_worker(
        self,
        deployment: InferenceWorkerDeploymentOptions,
    ) -> InferenceWorker:

        ## Call inference session builder based on path and other options
        ## Create the inference worker
        ## Return it

        raise NotImplementedError


class OnnxInferenceSessionBuilder:
    @classmethod
    async def build_inference_session(cls) -> ort.InferenceSession:
        raise NotImplementedError
