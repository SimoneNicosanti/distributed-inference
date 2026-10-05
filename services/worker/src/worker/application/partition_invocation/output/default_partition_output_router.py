from typing import override

from worker.application.partition_invocation.output.abc.partition_output_router import (
    PartitionOutputRouter,
)
from worker.application.ports.outbound.plan_store.service_inference_plan_store import (
    ServiceInferencePlanStore,
)
from worker.domain.partition.partition_invocation import (
    PartitionInvocationResult,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
    PartitionInvocationContributionContext,
)
from worker.domain.partition.tensor_bundle import TensorBundle


class DefaultPartitionOutputRouter(PartitionOutputRouter):
    def __init__(self, plan_store: ServiceInferencePlanStore):
        super().__init__()
        self._plan_store = plan_store

    @override
    async def route(
        self,
        contributions: list[PartitionInvocationContribution],
        partition_invocation_result: PartitionInvocationResult,
    ) -> list[PartitionInvocationContribution]:

        routed_contributions = []

        plan = await self._plan_store.get_service_inference_plan_by_version(
            contributions[0].plan_version
        )
        if plan is None:
            raise ValueError(
                f"Service inference plan for version {contributions[0].plan_version} not found"
            )

        partition_deployment = contributions[0].partition_deployment
        next_connections = plan.sub_model_next_connections[partition_deployment]

        for next_connection in next_connections:
            payload = TensorBundle(bundle={})
            for contribution in contributions:
                payload = payload.merge(
                    contribution.bundle.filter(next_connection.connection_tensors)
                )

            payload = payload.merge(
                partition_invocation_result.payload.filter(
                    next_connection.connection_tensors
                )
            )

            if set(payload.get_tensor_names()) != set(
                next_connection.connection_tensors
            ):
                raise ValueError(
                    f"Partition {partition_deployment} does not have all required tensors"
                )

            contribution_context = PartitionInvocationContributionContext(
                model_pass_context=contributions[0].model_pass_context,
                source_partition_invocation_context=partition_invocation_result.context,
                partition_deployment_id=next_connection.other_deployment,
            )

            routed_contribution = PartitionInvocationContribution(
                context=contribution_context, bundle=payload
            )

            routed_contributions.append(routed_contribution)

        return routed_contributions
