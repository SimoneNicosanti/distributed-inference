from typing import override

from shared.plan.hop import PlannedDestinationHopId
from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)
from worker.application.hop_invocation.abc.contribution_assembler import (
    ContributionAssembler,
    ContributionAssemblingError,
)
from worker.application.ports.outbound.plan_store.plan_store_errors import (
    NotExistentPlanError,
)
from worker.application.ports.outbound.plan_store.worker_sub_plan_readers import (
    PartitionExecutionPlanReader,
)
from worker.domain.context.hop_invocation_context import HopInvocationContext
from worker.domain.hop.hop_invocation import (
    HopInvocationRequest,
)
from worker.domain.hop.hop_invocation_contribution import (
    HopInvocationContribution,
)
from worker.domain.plan.partition_execution_plan import (
    NotExistentPartitionIOSchemeError,
)
from worker.domain.tensor.tensor_bundle import TensorBundle


class DefaultHopInvocationRequestAssembler(ContributionAssembler):
    def __init__(
        self, partition_execution_plan_reader: PartitionExecutionPlanReader
    ) -> None:
        super().__init__()
        self._partition_execution_plan_reader = partition_execution_plan_reader

    @override
    async def assemble(
        self,
        contributions: list[HopInvocationContribution],
    ) -> HopInvocationRequest[PlannedDestinationHopId]:
        if not contributions:
            raise ContributionAssemblingError(
                "Cannot assemble an empty contribution collection"
            )

        first = contributions[0]
        model_pass_context = first.model_pass_context
        destination_hop = first.destination_hop

        if any(
            contribution.model_pass_context != model_pass_context
            or contribution.destination_hop != destination_hop
            for contribution in contributions[1:]
        ):
            raise ContributionAssemblingError(
                "Contributions belong to different collections"
            )

        full_payload = TensorBundle(bundle={})
        for contribution in contributions:
            full_payload = full_payload.merge(contribution.bundle)

        match destination_hop:
            case PlannedPartitionReplicaId() as replica_id:
                try:
                    execution_plan = await self._partition_execution_plan_reader.get_partition_execution_plan_by_version(
                        first.plan_version
                    )
                    execution_scheme = execution_plan.get_scheme_by_partition_id(
                        replica_id.partition_id
                    )
                except (NotExistentPlanError, NotExistentPartitionIOSchemeError) as e:
                    raise ContributionAssemblingError(
                        "Plan for assembling does not exist or partition IO Scheme is missing"
                    ) from e

                missing_inputs = (
                    set(execution_scheme.inputs) - full_payload.get_tensor_names()
                )
                if missing_inputs:
                    raise ContributionAssemblingError(
                        f"Missing partition inputs: {missing_inputs}"
                    )
                payload = full_payload.filter(list(execution_scheme.inputs))

            case PlannedIngressReplicaId() | PlannedSinkReplicaId():
                payload = full_payload

            case hop:
                raise ContributionAssemblingError(
                    f"Unsupported destination hop: {hop!r}"
                )

        return HopInvocationRequest(
            context=HopInvocationContext(
                model_pass_context=model_pass_context,
                hop=destination_hop,
            ),
            payload=payload,
        )
