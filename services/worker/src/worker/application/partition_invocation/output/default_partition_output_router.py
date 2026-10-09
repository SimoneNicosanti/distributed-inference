from typing import override

from shared.tensor.tensor_bundle import TensorBundle
from worker.application.partition_invocation.output.abc.partition_output_router import (
    PartitionOutputRouter,
)
from worker.application.ports.outbound.plan_store.routing_plan_reader import (
    RoutingPlanReader,
)
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)
from worker.domain.partition.partition_invocation import (
    PartitionInvocationResult,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


class DefaultPartitionOutputRouter(PartitionOutputRouter):
    def __init__(
        self,
        routing_plan_reader: RoutingPlanReader,
    ) -> None:
        super().__init__()
        self._routing_plan_reader = routing_plan_reader

    @override
    async def route(
        self,
        contributions: list[PartitionInvocationContribution],
        partition_invocation_result: PartitionInvocationResult,
    ) -> list[PartitionInvocationContribution | ModelPassResultContribution]:
        plan_version = contributions[0].plan_version
        routing_plan = await self._routing_plan_reader.get_routing_plan_by_version(
            plan_version
        )
        if routing_plan is None:
            raise ValueError(f"Routing plan for version {plan_version} not found")

        current_replica_id = contributions[0].target_replica_id
        routing = routing_plan.get_routing_by_replica_id(current_replica_id)
        routed_contributions: list[
            PartitionInvocationContribution | ModelPassResultContribution
        ] = []

        available_tensor_names = set(
            partition_invocation_result.payload.get_tensor_names()
        )
        for contribution in contributions:
            available_tensor_names.update(contribution.bundle.get_tensor_names())

        output_tensors_by_replica = routing.route_output_tensors(available_tensor_names)

        for next_replica_id, routed_tensor_names in output_tensors_by_replica.items():
            tensor_names = list(routed_tensor_names)
            payload = TensorBundle(bundle={})

            for contribution in contributions:
                payload = payload.merge(contribution.bundle.filter(tensor_names))

            payload = payload.merge(
                partition_invocation_result.payload.filter(tensor_names)
            )

            if next_replica_id is None:
                routed_contributions.append(
                    ModelPassResultContribution(
                        model_pass_context=partition_invocation_result.model_pass_context,
                        bundle=payload,
                    )
                )
                continue

            routed_contributions.append(
                PartitionInvocationContribution(
                    source_invocation_context=partition_invocation_result.context,
                    target_replica_id=next_replica_id,
                    bundle=payload,
                )
            )

        return routed_contributions
