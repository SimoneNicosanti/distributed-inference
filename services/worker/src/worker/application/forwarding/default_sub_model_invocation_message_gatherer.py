from typing import override

from integration.plan.plan import ServiceInferencePlan
from worker.application.deployment.contracts.service_inference_plan_preparer import (
    ServiceInferencePlanPreparer,
)
from worker.application.forwarding.contracts.gathering.gather_key import (
    GatherKey,
)
from worker.application.forwarding.contracts.gathering.sub_model_invocation_message_gatherer import (
    SubModelInvocationMessageGatherer,
)
from worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from worker.application.ports.outbound.sub_model_invocation_message_store import (
    SubModelInvocationMessageGatheringStore,
)
from worker.domain.sub_model.invocation.sub_model_invocation_context import (
    SubModelInvocationId,
)
from worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
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
        self._plan_store: ServiceInferencePlanStore = service_inference_plan_store
        self._gathering_store: SubModelInvocationMessageGatheringStore = gathering_store
        self._gathering_key_to_ids: dict[GatherKey, SubModelInvocationId] = {}

    @override
    async def gather_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> tuple[list[SubModelInvocationMessage] | None, SubModelInvocationId]:

        plan_version = sub_model_invocation_message.plan_version
        plan = await self._plan_store.get_service_inference_plan_by_version(
            plan_version
        )
        if plan is None:
            raise ValueError(
                f"Service inference plan for version {plan_version} not found"
            )

        gather_key = await self._gathering_store.put_sub_model_invocation_message(
            sub_model_invocation_message
        )
        if gather_key not in self._gathering_key_to_ids:
            sub_model_invocation_id = SubModelInvocationId()
            self._gathering_key_to_ids[gather_key] = sub_model_invocation_id
        sub_model_invocation_id = self._gathering_key_to_ids[gather_key]

        collected_invocation_messages = (
            await self._gathering_store.get_by_gathering_key(gather_key)
        )

        arrived_all = self._check_arrived_all(collected_invocation_messages, plan)

        if arrived_all:
            # sub_model_invocation_request = self._build_sub_model_invocation_request(
            #     collected_invocation_messages, sub_model_invocation_id
            # )
            await self._clear_all_by_gathering_key(gather_key)
            return collected_invocation_messages, sub_model_invocation_id
        else:
            return None, sub_model_invocation_id

    async def _clear_all_by_gathering_key(self, gather_key: GatherKey) -> None:
        self._gathering_key_to_ids.pop(gather_key)
        await self._gathering_store.delete_by_gathering_key(gather_key)

    def _check_arrived_all(
        self,
        all_sub_model_invocation_messages: list[SubModelInvocationMessage],
        service_inference_plan: ServiceInferencePlan,
    ) -> bool:

        sub_model_id = all_sub_model_invocation_messages[0].sub_model_id

        arrived_tensors: set[str] = set()
        for msg in all_sub_model_invocation_messages:
            arrived_tensors.update(msg.payload.bundle.keys())

        execution_scheme = service_inference_plan.sub_model_execution_schemes[
            sub_model_id
        ]
        skip_scheme = service_inference_plan.sub_model_skip_schemes[sub_model_id]

        ## TODO: To do a better check, we should check that the tensors arrived from the expected sources according with the topology expressed in the plan
        ## TODO : check this in case of model split changes
        return set(arrived_tensors) == set(execution_scheme.inputs).union(
            skip_scheme.skip_tensors
        )

    @override
    async def prepare_service_inference_plan(
        self, service_inference_plan: ServiceInferencePlan
    ) -> None:
        ## No need to do anything, we just check the plans in the store
        pass
