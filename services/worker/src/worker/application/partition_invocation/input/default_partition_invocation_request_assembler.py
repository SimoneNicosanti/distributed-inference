from typing import override

from worker.application.partition_invocation.input.abc.partition_invocation_request_assembler import (
    PartitionInvocationRequestAssembler,
)
from worker.application.ports.outbound.service_inference_plan_store import (
    ServiceInferencePlanStore,
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
from worker.domain.partition.tensor_bundle import TensorBundle


class DefaultPartitionInvocationRequestAssembler(PartitionInvocationRequestAssembler):
    def __init__(self, plan_store: ServiceInferencePlanStore):
        super().__init__()
        self._plan_store = plan_store

    ## Builds a complete partition invocation request from collected contributions.
    @override
    async def assemble(
        self,
        contributions: list[PartitionInvocationContribution],
    ) -> PartitionInvocationRequest:

        plan = await self._plan_store.get_service_inference_plan_by_version(
            contributions[0].plan_version
        )
        if plan is None:
            raise ValueError(
                f"Service inference plan for version {contributions[0].plan_version} not found"
            )

        partition_id = contributions[0].partition_id
        execution_scheme = plan.sub_model_execution_schemes[partition_id]

        full_payload = TensorBundle(bundle={})
        for contribution in contributions:
            full_payload = full_payload.merge(
                contribution.bundle.filter(execution_scheme.inputs)
            )

        model_pass_context = contributions[0].model_pass_context
        partition_deployment_id = contributions[0].partition_deployment

        return PartitionInvocationRequest(
            context=PartitionInvocationContext(
                model_pass_context=model_pass_context,
                partition_deployment_id=partition_deployment_id,
            ),
            payload=full_payload,
        )
