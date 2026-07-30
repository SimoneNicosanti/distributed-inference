from typing import Tuple, override

from distributed_inference.domain.plan import ServiceInferencePlan
from distributed_inference.worker.application.deployment.contracts.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from distributed_inference.worker.application.forwarding.contracts.gathering.gather_key import (
    GatherKey,
)
from distributed_inference.worker.application.forwarding.contracts.gathering.sub_model_invocation_message_gatherer import (
    SubModelInvocationMessageGatherer,
)
from distributed_inference.worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from distributed_inference.worker.application.ports.outbound.sub_model_invocation_message_store import (
    SubModelInvocationMessageGatheringStore,
)
from distributed_inference.worker.domain.sub_model.invocation.sub_model_invocation_context import (
    SubModelInvocationId,
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
        gathering_store: SubModelInvocationMessageGatheringStore,
    ):
        super().__init__()
        self._service_inference_plan_store: ServiceInferencePlanStore = (
            service_inference_plan_store
        )
        self._gathering_store: SubModelInvocationMessageGatheringStore = gathering_store
        self._gathering_key_to_ids: dict[GatherKey, SubModelInvocationId] = {}

    @override
    async def gather_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> Tuple[SubModelInvocationRequest | None, SubModelInvocationId]:

        plan_version = sub_model_invocation_message.get_plan_version()
        service_inference_plan = await self._service_inference_plan_store.get_service_inference_plan_by_version(
            plan_version
        )
        if service_inference_plan is None:
            raise ValueError(
                f"Service inference plan for version {plan_version} not found"
            )

        gather_key = await self._gathering_store.put_sub_model_invocation_message(
            sub_model_invocation_message
        )
        if gather_key not in self._gathering_key_to_ids:
            sub_model_incoation_id = SubModelInvocationId()
            self._gathering_key_to_ids[gather_key] = sub_model_incoation_id
        sub_model_invocation_id = self._gathering_key_to_ids[gather_key]

        all_sub_model_invocation_messages = await self._gathering_store.get_all_sub_model_invocation_message_by_gathering_key(
            gather_key
        )

        arrived_all = self._check_arrived_all(
            all_sub_model_invocation_messages, service_inference_plan
        )

        if arrived_all:
            sub_model_invocation_request = self._build_sub_model_invocation_request(
                all_sub_model_invocation_messages, service_inference_plan
            )
            return sub_model_invocation_request, sub_model_invocation_id
        else:
            return None, sub_model_invocation_id

    def _check_arrived_all(
        self,
        all_sub_model_invocation_messages: list[SubModelInvocationMessage],
        service_inference_plan: ServiceInferencePlan,
    ) -> bool:
        raise NotImplementedError

    def _build_sub_model_invocation_request(
        self,
        all_sub_model_invocation_messages: list[SubModelInvocationMessage],
        service_inference_plan: ServiceInferencePlan,
    ) -> SubModelInvocationRequest:
        raise NotImplementedError

    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        raise NotImplementedError
