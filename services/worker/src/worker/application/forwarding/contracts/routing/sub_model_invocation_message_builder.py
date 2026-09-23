from worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
    SubModelInvocationMessageContext,
)
from worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationResponse,
)
from worker.domain.tensor.tensor import TensorBundle


class SubModelInvocationMessageBuilder:
    def __init__(self, plan_store: ServiceInferencePlanStore):
        super().__init__()
        self._plan_store = plan_store

    async def build_sub_model_invocation_message_from_context(
        self,
        sub_model_invocation_messages: list[SubModelInvocationMessage],
        sub_model_inference_response: SubModelInvocationResponse,
    ) -> list[SubModelInvocationMessage]:

        invocation_messages = []

        plan = await self._plan_store.get_service_inference_plan_by_version(
            sub_model_invocation_messages[0].plan_version
        )
        if plan is None:
            raise ValueError(
                f"Service inference plan for version {sub_model_invocation_messages[0].plan_version} not found"
            )

        sub_model_deployment = sub_model_invocation_messages[0].sub_model_deployment_id
        next_connections = plan.sub_model_next_connections[sub_model_deployment]

        for next_connection in next_connections:
            payload = TensorBundle(bundle={})
            for msg in sub_model_invocation_messages:
                payload = payload.merge(
                    msg.payload.filter(next_connection.connection_tensors)
                )

            payload = payload.merge(
                sub_model_inference_response.payload.filter(
                    next_connection.connection_tensors
                )
            )

            if set(payload.get_tensor_names()) != set(
                next_connection.connection_tensors
            ):
                raise ValueError(
                    f"Sub-model {sub_model_deployment} does not have all the tensors it needs to pass"
                )

            sub_model_invocation_message_ctx = SubModelInvocationMessageContext(
                model_pass_context=sub_model_invocation_messages[0].model_pass_context,
                source_sub_model_invocation_context=sub_model_inference_response.context,
                sub_model_deployment_id=next_connection.other_deployment,
            )

            invocation_message = SubModelInvocationMessage(
                context=sub_model_invocation_message_ctx, payload=payload
            )

            invocation_messages.append(invocation_message)

        return invocation_messages
