from typing import override

from worker.application.partition_execution.abc.partition_execution_coordinator import (
    PartitionExecutionCoordinator,
)
from worker.application.partition_invocation.abc.partition_invocation_completion_registry import (
    PartitionInvocationCompletionRegistry,
)
from worker.application.partition_invocation.input.abc.partition_invocation_contribution_collector import (
    PartitionInvocationContributionCollector,
)
from worker.application.partition_invocation.output.partition_invocation_request_assembler import (
    PartitionInvocationRequestAssembler,
)
from worker.application.partition_invocation.output.partition_output_router import (
    PartitionOutputRouter,
)
from worker.application.ports.inbound.partition_invocation_coordinator import (
    PartitionInvocationCoordinator,
)
from worker.application.ports.outbound.partition_invocation_contribution_sender import (
    PartitionInvocationContributionSender,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
    PartitionInvocationContributionAck,
)


class DefaultPartitionInvocationCoordinator(PartitionInvocationCoordinator):
    def __init__(
        self,
        contribution_collector: PartitionInvocationContributionCollector,
        partition_execution_coordinator: PartitionExecutionCoordinator,
        output_router: PartitionOutputRouter,
        contribution_sender: PartitionInvocationContributionSender,
        completion_registry: PartitionInvocationCompletionRegistry,
        request_assembler: PartitionInvocationRequestAssembler,
    ):
        self._contribution_collector = contribution_collector
        self._partition_execution_coordinator = partition_execution_coordinator
        self._output_router = output_router
        self._contribution_sender = contribution_sender
        self._completion_registry = completion_registry
        self._request_assembler = request_assembler

    @override
    async def process_partition_invocation_contribution(
        self, contribution: PartitionInvocationContribution
    ) -> PartitionInvocationContributionAck:

        (
            collected_contributions,
            partition_invocation_id,
        ) = await self._contribution_collector.collect(contribution)
        if collected_contributions is None:
            await self._completion_registry.wait_for_partition_invocation_completion(
                partition_invocation_id
            )
            return PartitionInvocationContributionAck(context=contribution.context)

        partition_invocation_request = await self._request_assembler.assemble(
            collected_contributions, partition_invocation_id
        )
        partition_invocation_result = await self._partition_execution_coordinator.process_partition_invocation_request(
            partition_invocation_request
        )

        routed_contributions = await self._output_router.route(
            collected_contributions, partition_invocation_result
        )

        for routed_contribution in routed_contributions:
            self._contribution_sender.send(routed_contribution)

        await self._completion_registry.register_partition_invocation_success(
            partition_invocation_id
        )

        return PartitionInvocationContributionAck(context=contribution.context)
