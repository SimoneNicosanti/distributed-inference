from typing import override

from shared.context.model_pass_context import ModelPassId
from shared.model.keys import TensorKey
from worker.application.contribution.abc.contribution_gathering_policy import (
    ContributionGatheringPolicy,
)
from worker.application.ports.outbound.plan_store.plan_store_reader import (
    PlanStoreReader,
)
from worker.application.sink.model_pass_result_collection import (
    ModelPassResultCollection,
)
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)
from worker.domain.plan.sink.sink_plan import SinkPlan


class ModelPassResultContributionGatheringPolicy(
    ContributionGatheringPolicy[
        ModelPassId,
        ModelPassResultContribution,
        ModelPassResultCollection,
    ]
):
    def __init__(self, sink_plan_reader: PlanStoreReader[SinkPlan]) -> None:
        self._sink_plan_reader = sink_plan_reader

    @override
    async def key_of(self, contribution: ModelPassResultContribution) -> ModelPassId:
        return contribution.model_pass_id

    @override
    async def validate(
        self,
        other_contributions: list[ModelPassResultContribution],
        new_contribution: ModelPassResultContribution,
    ) -> None:
        for contribution in other_contributions:
            if contribution.model_pass_context != new_contribution.model_pass_context:
                raise ValueError(
                    "Contributions for the same model pass must share the same context"
                )

        expected_tensor_keys = await self._get_expected_tensor_keys(new_contribution)
        new_tensor_keys = new_contribution.bundle.get_tensor_names()
        if not new_tensor_keys:
            raise ValueError("A model pass result contribution must not be empty")

        unexpected_tensor_keys = new_tensor_keys - expected_tensor_keys
        if unexpected_tensor_keys:
            raise ValueError(
                f"Unexpected model pass result tensors: {unexpected_tensor_keys}"
            )

        collected_tensor_keys = self._get_collected_tensor_keys(other_contributions)
        duplicate_tensor_keys = collected_tensor_keys & new_tensor_keys
        if duplicate_tensor_keys:
            raise ValueError(
                f"Duplicate model pass result tensors: {duplicate_tensor_keys}"
            )

    @override
    async def is_ready(
        self,
        contributions: list[ModelPassResultContribution],
    ) -> bool:
        if not contributions:
            return False

        expected_tensor_keys = await self._get_expected_tensor_keys(contributions[0])
        collected_tensor_keys = self._get_collected_tensor_keys(contributions)

        unexpected_tensor_keys = collected_tensor_keys - expected_tensor_keys
        if unexpected_tensor_keys:
            raise ValueError(
                f"Unexpected model pass result tensors: {unexpected_tensor_keys}"
            )

        return collected_tensor_keys == expected_tensor_keys

    @override
    async def gather(
        self,
        contributions: list[ModelPassResultContribution],
    ) -> ModelPassResultCollection:
        if not contributions:
            raise ValueError("Cannot gather an empty model pass result collection")

        return ModelPassResultCollection(
            key=contributions[0].model_pass_id,
            contributions=tuple(contributions),
        )

    async def _get_expected_tensor_keys(
        self,
        contribution: ModelPassResultContribution,
    ) -> set[TensorKey]:
        model_pass_context = contribution.model_pass_context
        sink_plan = await self._sink_plan_reader.get_plan_by_version(
            model_pass_context.plan_version
        )
        flow_id = model_pass_context.model_invocation_context.flow_id
        return set(sink_plan.get_result_tensors_by_flow_id(flow_id))

    @staticmethod
    def _get_collected_tensor_keys(
        contributions: list[ModelPassResultContribution],
    ) -> set[TensorKey]:
        tensor_keys: set[TensorKey] = set()
        for contribution in contributions:
            tensor_keys.update(contribution.bundle.get_tensor_names())

        return tensor_keys
