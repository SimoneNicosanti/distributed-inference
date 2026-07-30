from typing import override

from distributed_inference.worker.application.forwarding.contracts.sub_model_inference_message_gatherer import (
    SubModelInferenceMessageGatherer,
)
from distributed_inference.worker.application.forwarding.contracts.sub_model_inference_response_router import (
    SubModelInferenceResponseRouter,
)
from distributed_inference.worker.application.sub_model_execution.contracts.sub_model_execution_coordinator import (
    SubModelExecutionCoordinator,
)
from distributed_inference.worker.application.ports.inbound.inference_flow_coordinator import (
    InferenceFlowCoordinator,
)
from distributed_inference.worker.application.ports.outbound.sub_model_inference_message_forwarder import (
    SubModelInferenceMessageForwarder,
)
from distributed_inference.worker.domain.sub_model_inference_message import (
    SubModelInferenceMessage,
)
from distributed_inference.worker.domain.sub_model_inference_request_response import (
    SubModelInferenceResponse,
)
from distributed_inference.worker.domain.sub_model_output_route_instruction import (
    SubModelOutputRouteInstruction,
)


class DefaultInferenceFlowCoordinator(InferenceFlowCoordinator):
    def __init__(
        self,
        sub_model_inference_message_gatherer: SubModelInferenceMessageGatherer,
        sub_model_execution_coordinator: SubModelExecutionCoordinator,
        sub_model_inference_response_router: SubModelInferenceResponseRouter,
        sub_model_inference_message_forwarder: SubModelInferenceMessageForwarder,
    ):
        self._sub_model_inference_message_gatherer = (
            sub_model_inference_message_gatherer
        )
        self._sub_model_execution_coordinator = sub_model_execution_coordinator
        self._sub_model_inference_response_router = sub_model_inference_response_router
        self._sub_model_inference_message_forwarder = (
            sub_model_inference_message_forwarder
        )

    @override
    async def process_sub_model_inference_message(
        self, sub_model_inference_message: SubModelInferenceMessage
    ) -> None:

        inference_request = await self._sub_model_inference_message_gatherer.gather_sub_model_inference_message(
            sub_model_inference_message
        )
        if inference_request is None:
            ## Inference request could not be gathered
            ## We need to wait for other inference messages to come
            return None

        inference_response = await self._sub_model_execution_coordinator.process_sub_model_inference_request(
            inference_request
        )

        route_instructions = await self._sub_model_inference_response_router.route_sub_model_inference_response(
            inference_response
        )

        for route_instruction in route_instructions:
            ## TODO: Build inference message for the next server
            next_inference_message = self.__build_next_inference_message(
                inference_response, route_instruction
            )
            self._sub_model_inference_message_forwarder.forward_sub_model_inference_message(
                next_inference_message, route_instruction
            )

        return None

    @classmethod
    def __build_next_inference_message(
        cls,
        inference_response: SubModelInferenceResponse,
        route_instruction: SubModelOutputRouteInstruction,
    ) -> SubModelInferenceMessage:
        raise NotImplementedError
