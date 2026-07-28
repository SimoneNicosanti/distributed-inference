from abc import abstractmethod

from distributed_inference.application.worker.domain.inference_flow import (
    InferenceMessage,
    InferenceRequest,
)


class InferenceMessageGatherer:
    @abstractmethod
    async def gather_inference_message(
        self, inference_message: InferenceMessage
    ) -> InferenceRequest | None: ...
