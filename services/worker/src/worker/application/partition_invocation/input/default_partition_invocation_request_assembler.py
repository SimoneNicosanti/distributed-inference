from typing import override

from worker.application.contribution.abc.contribution_assembler import (
    ContributionAssembler,
)
from worker.application.ports.outbound.plan_store.worker_sub_plan_readers import (
    PartitionExecutionPlanReader,
)
from worker.domain.context.partition_invocation_context import (
    PartitionInvocationContext,
)
from worker.domain.partition.partition_invocation import (
    PartitionInvocationRequest,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)
from worker.domain.tensor.tensor_bundle import TensorBundle


class DefaultPartitionInvocationRequestAssembler(
    ContributionAssembler[PartitionInvocationContribution, PartitionInvocationRequest]
):
    def __init__(
        self, partition_execution_plan_reader: PartitionExecutionPlanReader
    ) -> None:
        super().__init__()
        self._partition_execution_plan_reader = partition_execution_plan_reader

    @override
    async def assemble(
        self,
        contributions: list[PartitionInvocationContribution],
    ) -> PartitionInvocationRequest:
        plan_version = contributions[0].plan_version
        execution_plan = await self._partition_execution_plan_reader.get_partition_execution_plan_by_version(
            plan_version
        )
        partition_id = contributions[0].partition_id
        execution_scheme = execution_plan.get_scheme_by_partition_id(partition_id)

        full_payload = TensorBundle(bundle={})
        for contribution in contributions:
            full_payload = full_payload.merge(
                contribution.bundle.filter(list(execution_scheme.inputs))
            )

        model_pass_context = contributions[0].model_pass_context
        replica_id = contributions[0].target_replica_id

        return PartitionInvocationRequest(
            context=PartitionInvocationContext(
                model_pass_context=model_pass_context,
                partition_replica_id=replica_id,
            ),
            payload=full_payload,
        )
