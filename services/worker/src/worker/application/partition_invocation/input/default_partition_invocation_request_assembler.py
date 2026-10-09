from typing import override

from shared.tensor.tensor_bundle import TensorBundle
from worker.application.partition_invocation.input.abc.partition_invocation_request_assembler import (
    PartitionInvocationRequestAssembler,
)
from worker.application.ports.outbound.plan_store.partition_execution_plan_reader import (
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


class DefaultPartitionInvocationRequestAssembler(PartitionInvocationRequestAssembler):
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
        if execution_plan is None:
            raise ValueError(
                f"Partition execution plan for version {plan_version} not found"
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
