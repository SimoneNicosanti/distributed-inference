from typing import cast, override

from worker.application.hop_contribution_snd.abc.hop_output_router import (
    HopOutputRouter,
)
from worker.application.ports.outbound.plan_store.worker_sub_plan_readers import (
    HopRoutingPlanReader,
)
from worker.domain.context.hop_invocation_context import HopInvocationContext
from worker.domain.hop.hop import InvocableHop, SinkHop, SourceHop
from worker.domain.hop.hop_invocation import (
    HopInvocationResult,
)
from worker.domain.hop.hop_invocation_contribution import (
    HopInvocationContribution,
)
from worker.domain.plan.hop_routing_plan import HopAddress
from worker.domain.tensor.tensor_bundle import TensorBundle


class DefaultHopOutputRouter(HopOutputRouter):
    def __init__(
        self,
        routing_plan_reader: HopRoutingPlanReader,
    ) -> None:
        super().__init__()
        self._routing_plan_reader = routing_plan_reader

    @override
    async def route(
        self,
        incoming_contributions: list[HopInvocationContribution],
        hop_invocation_result: HopInvocationResult[InvocableHop],
    ) -> list[HopInvocationContribution]:
        model_pass_context = hop_invocation_result.model_pass_context
        flow_id = hop_invocation_result.flow_id

        routing_plan = await self._routing_plan_reader.get_hop_routing_plan_by_version(
            model_pass_context.plan_version
        )
        source_address = HopAddress(
            flow_id=flow_id,
            hop=hop_invocation_result.context.hop,
        )
        routing = routing_plan.get_routings_for_hop_address(source_address)

        available_payload = TensorBundle(bundle={})
        for contribution in incoming_contributions:
            available_payload = available_payload.merge(contribution.bundle)
        available_payload = available_payload.merge(hop_invocation_result.payload)

        output_tensors_by_destination = routing.route_output_tensors(
            available_payload.get_tensor_names()
        )

        if isinstance(hop_invocation_result.context.hop, SinkHop):
            if output_tensors_by_destination:
                raise ValueError("Sink hop cannot have output routes")
            return []

        source_context = cast(
            HopInvocationContext[SourceHop], hop_invocation_result.context
        )
        routed_contributions: list[HopInvocationContribution] = []

        for destination, tensor_names in output_tensors_by_destination.items():
            if destination.flow_id != flow_id:
                raise ValueError(
                    "Cross-flow routing requires an explicit destination context"
                )

            routed_contributions.append(
                HopInvocationContribution(
                    source_context=source_context,
                    destination_hop=destination.hop,
                    bundle=available_payload.filter(list(tensor_names)),
                )
            )

        return routed_contributions
