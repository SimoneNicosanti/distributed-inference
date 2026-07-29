from typing import override

from distributed_inference.worker.application.forwarding.contracts.inference_message_gatherer import (
    InferenceMessageGatherer,
)
from distributed_inference.worker.application.forwarding.contracts.inference_response_router import (
    InferenceResponseRouter,
)
from distributed_inference.worker.application.inference.contracts.inference_worker_coordinator import (
    InferenceWorkerCoordinator,
)
from distributed_inference.worker.application.ports.inbound.execution_coordinator import (
    ExecutionCoordinator,
)
from distributed_inference.worker.application.ports.outbound.inference_message_forwarder import (
    InferenceMessageForwarder,
)
from distributed_inference.worker.domain.inference_flow import (
    InferenceMessage,
    InferenceResponse,
)
from distributed_inference.worker.domain.route_instruction import (
    RouteInstruction,
)


class DefaultExecutionCoordinator(ExecutionCoordinator):
    def __init__(
        self,
        inference_message_gatherer: InferenceMessageGatherer,
        inference_coordinator: InferenceWorkerCoordinator,
        inference_response_router: InferenceResponseRouter,
        inference_message_forwarder: InferenceMessageForwarder,
    ):
        self._inference_message_gatherer = inference_message_gatherer
        self._inference_coordinator = inference_coordinator
        self._inference_response_router = inference_response_router
        self._inference_message_forwarder = inference_message_forwarder

    @override
    async def process_inference_message(
        self, inference_message: InferenceMessage
    ) -> None:

        inference_request = (
            await self._inference_message_gatherer.gather_inference_message(
                inference_message
            )
        )
        if inference_request is None:
            ## Inference request could not be gathered
            ## We need to wait for other inference messages to come
            return None

        inference_response = (
            await self._inference_coordinator.process_inference_request(
                inference_request
            )
        )

        route_instructions = (
            await self._inference_response_router.route_inference_response(
                inference_response
            )
        )

        for route_instruction in route_instructions:
            ## TODO: Build inference message for the next server
            next_inference_message = self.__build_next_inference_message(
                inference_response, route_instruction
            )
            self._inference_message_forwarder.forward_inference_message(
                next_inference_message, route_instruction
            )

        return None

    @classmethod
    def __build_next_inference_message(
        cls, inference_response: InferenceResponse, route_instruction: RouteInstruction
    ) -> InferenceMessage:
        raise NotImplementedError
