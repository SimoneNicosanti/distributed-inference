from typing import override

from distributed_inference.domain.plan import ServiceInferencePlan
from distributed_inference.worker.application.deployment.contracts.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from distributed_inference.worker.application.forwarding.contracts.sub_model_invocation_message_gatherer import (
    SubModelInvocationMessageGatherer,
)
from distributed_inference.worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from distributed_inference.worker.application.ports.outbound.sub_model_invocation_message_store import (
    SubModelInvocationMessageStore,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_request_response import (
    SubModelInvocationRequest,
)


class DefaultSubModelInvocationMessageGatherer(
    SubModelInvocationMessageGatherer, ServiceInferencePlanPreparer
):
    def __init__(
        self,
        service_inference_plan_store: ServiceInferencePlanStore,
        gathering_store: SubModelInvocationMessageStore,
    ):
        super().__init__()
        self._service_inference_plan_store: ServiceInferencePlanStore = (
            service_inference_plan_store
        )
        self._gathering_store: SubModelInvocationMessageStore = gathering_store

    @override
    async def gather_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> SubModelInvocationRequest | None:

        plan_version = sub_model_invocation_message.get_plan_version()
        service_inference_plan = await self._service_inference_plan_store.get_service_inference_plan_by_version(
            plan_version
        )
        if service_inference_plan is None:
            raise ValueError(
                f"Service inference plan for version {plan_version} not found"
            )

        return None

    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        raise NotImplementedError
