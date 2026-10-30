from typing import override

from worker.application.ingress.abc.ingress_input_router import IngressInputRouter
from worker.domain.context.ingress_invocation_context import IngressInvocationContext
from worker.domain.model_pass.model_pass_request import ModelPassRequest
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)
from worker.domain.plan.ingress.flow_ingress_plan import FlowIngressPlan


class DefaultIngressInputRouter(IngressInputRouter):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @override
    async def route(
        self,
        model_pass_request: ModelPassRequest,
        ingress_context: IngressInvocationContext,
        flow_plan: FlowIngressPlan,
    ) -> list[PartitionInvocationContribution | ModelPassResultContribution]:
        available_tensor_names = model_pass_request.bundle.get_tensor_names()
        routes = flow_plan.route_input_tensors(available_tensor_names)

        return [
            PartitionInvocationContribution(
                source_hop_invocation_context=ingress_context,
                target_replica_id=replica_id,
                bundle=model_pass_request.bundle.filter(list(tensor_names)),
            )
            for replica_id, tensor_names in routes.items()
        ]
