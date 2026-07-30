from typing import override

from distributed_inference.worker.application.flow.contracts.sub_model_invocation_completition_registry import (
    SubModelInvocationCompletitionRegistry,
)
from distributed_inference.worker.application.forwarding.contracts.gathering.sub_model_invocation_message_gatherer import (
    SubModelInvocationMessageGatherer,
)
from distributed_inference.worker.application.forwarding.contracts.routing.sub_model_invocation_response_router import (
    SubModelInvocationResponseRouter,
)
from distributed_inference.worker.application.ports.inbound.inference_flow_coordinator import (
    InferenceFlowCoordinator,
)
from distributed_inference.worker.application.ports.outbound.sub_model_invocation_message_forwarder import (
    SubModelInvocationMessageForwarder,
)
from distributed_inference.worker.application.sub_model_execution.contracts.sub_model_execution_coordinator import (
    SubModelExecutionCoordinator,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationAck,
    SubModelInvocationMessage,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationResponse,
)
from distributed_inference.worker.domain.sub_model.route.sub_model_invocation_response_route import (
    SubModelInvocationResponseRoute,
)


class DefaultInferenceFlowCoordinator(InferenceFlowCoordinator):
    def __init__(
        self,
        sub_model_invocation_message_gatherer: SubModelInvocationMessageGatherer,
        sub_model_execution_coordinator: SubModelExecutionCoordinator,
        sub_model_invocation_response_router: SubModelInvocationResponseRouter,
        sub_model_invocation_message_forwarder: SubModelInvocationMessageForwarder,
        completition_registry: SubModelInvocationCompletitionRegistry,
    ):
        self._sub_model_invocation_message_gatherer = (
            sub_model_invocation_message_gatherer
        )
        self._sub_model_execution_coordinator = sub_model_execution_coordinator
        self._sub_model_invocation_response_router = (
            sub_model_invocation_response_router
        )
        self._sub_model_invocation_message_forwarder = (
            sub_model_invocation_message_forwarder
        )
        self._completition_registry = completition_registry

    @override
    async def process_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> SubModelInvocationAck:

        (
            sub_model_inference_request,
            sub_model_invocation_id,
        ) = await self._sub_model_invocation_message_gatherer.gather_sub_model_invocation_message(
            sub_model_invocation_message
        )
        if sub_model_inference_request is None:
            ## Inference request could not be gathered
            ## We need to wait for other inference messages to come
            ## We wait to inference completition event to return a positive ack to the sender

            ## TODO: We should manage the error here
            await (
                self._completition_registry.wait_for_sub_model_invocation_completition(
                    sub_model_invocation_id
                )
            )
            return SubModelInvocationAck(context=sub_model_invocation_message.context)

        sub_model_inference_response = await self._sub_model_execution_coordinator.process_sub_model_invocation_request(
            sub_model_inference_request
        )

        route_instructions = await self._sub_model_invocation_response_router.route_sub_model_invocation_response(
            sub_model_inference_response
        )

        for route_instruction in route_instructions:
            ## TODO: Build inference message for the next server
            next_inference_message = self.__build_next_sub_model_invocation_message(
                sub_model_inference_response, route_instruction
            )
            self._sub_model_invocation_message_forwarder.forward_sub_model_invocation_message(
                next_inference_message, route_instruction
            )

        ## We register the completition event on the registry
        await self._completition_registry.register_sub_model_invocation_success(
            sub_model_invocation_id
        )

        ## We return the ack of the coroutine that has materially done the execution
        return SubModelInvocationAck(context=sub_model_invocation_message.context)

    @classmethod
    def __build_next_sub_model_invocation_message(
        cls,
        inference_response: SubModelInvocationResponse,
        route_instruction: SubModelInvocationResponseRoute,
    ) -> SubModelInvocationMessage:
        raise NotImplementedError
