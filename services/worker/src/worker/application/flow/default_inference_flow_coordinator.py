from typing import override

from worker.application.flow.contracts.sub_model_invocation_completition_registry import (
    SubModelInvocationCompletitionRegistry,
)
from worker.application.forwarding.contracts.gathering.sub_model_invocation_message_gatherer import (
    SubModelInvocationMessageGatherer,
)
from worker.application.forwarding.contracts.routing.sub_model_invocation_message_builder import (
    SubModelInvocationMessageBuilder,
)
from worker.application.forwarding.contracts.routing.sub_model_invocation_request_builder import (
    SubModelInvocationRequestBuilder,
)
from worker.application.ports.inbound.inference_flow_coordinator import (
    InferenceFlowCoordinator,
)
from worker.application.ports.outbound.sub_model_invocation_message_forwarder import (
    SubModelInvocationMessageForwarder,
)
from worker.application.sub_model_execution.contracts.sub_model_execution_coordinator import (
    SubModelExecutionCoordinator,
)
from worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationAck,
    SubModelInvocationMessage,
)


class DefaultInferenceFlowCoordinator(InferenceFlowCoordinator):
    def __init__(
        self,
        sub_model_invocation_message_gatherer: SubModelInvocationMessageGatherer,
        sub_model_execution_coordinator: SubModelExecutionCoordinator,
        sub_model_invocation_response_router: SubModelInvocationMessageBuilder,
        sub_model_invocation_message_forwarder: SubModelInvocationMessageForwarder,
        completition_registry: SubModelInvocationCompletitionRegistry,
        sub_model_invocation_request_builder: SubModelInvocationRequestBuilder,
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

        self._sub_model_invocation_request_builder = (
            sub_model_invocation_request_builder
        )

    @override
    async def process_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> SubModelInvocationAck:

        (
            sub_model_invocation_messages,
            sub_model_invocation_id,
        ) = await self._sub_model_invocation_message_gatherer.gather_sub_model_invocation_message(
            sub_model_invocation_message
        )
        if sub_model_invocation_messages is None:
            ## Inference request could not be gathered
            ## We need to wait for other inference messages to come
            ## We wait to inference completition event to return a positive ack to the sender

            ## TODO: We should manage the error here
            ## The happy path should work though
            await (
                self._completition_registry.wait_for_sub_model_invocation_completition(
                    sub_model_invocation_id
                )
            )
            return SubModelInvocationAck(context=sub_model_invocation_message.context)

        sub_model_invocation_request = await self._sub_model_invocation_request_builder.build_sub_model_invocation_request(
            sub_model_invocation_messages, sub_model_invocation_id
        )
        sub_model_invocation_response = await self._sub_model_execution_coordinator.process_sub_model_invocation_request(
            sub_model_invocation_request
        )

        next_sub_model_invocation_messages = await self._sub_model_invocation_response_router.build_sub_model_invocation_message_from_context(
            sub_model_invocation_messages, sub_model_invocation_response
        )

        for next_sub_model_invocation_message in next_sub_model_invocation_messages:
            self._sub_model_invocation_message_forwarder.forward_sub_model_invocation_message(
                next_sub_model_invocation_message
            )

        ## We register the completition event on the registry
        await self._completition_registry.register_sub_model_invocation_success(
            sub_model_invocation_id
        )

        ## We return the ack of the coroutine that has materially done the execution
        return SubModelInvocationAck(context=sub_model_invocation_message.context)
