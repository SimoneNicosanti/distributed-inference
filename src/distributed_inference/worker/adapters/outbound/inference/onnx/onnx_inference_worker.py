import gc
from typing import override

import onnxruntime as ort

from distributed_inference.worker.application.ports.outbound.inference_worker import (
    InferenceWorker,
)
from distributed_inference.worker.domain.inference_run import (
    InferenceInput,
    InferenceOutput,
)


class OnnxInferenceWorker(InferenceWorker):
    def __init__(self, inference_session: ort.InferenceSession) -> None:
        ## Build session to be reused across requests
        self.inference_session: ort.InferenceSession = inference_session

    @override
    async def close(self) -> None:
        del self.inference_session
        gc.collect()
        pass

    @override
    async def process_inference_input(
        self, inference_input: InferenceInput
    ) -> InferenceOutput:

        ## Three ways to run inference
        ## Using a thread in the same process
        ## Using a separate process
        ## Using a separate process in a process pool

        ## await asyncio.to_thread()
        ## await asyncio.create_subprocess_exec
        ## loop = asyncio.get_running_loop() + await loop.run_in_executor

        ## We just need a way to call await, otherwise we would block the main loop

        raise NotImplementedError
