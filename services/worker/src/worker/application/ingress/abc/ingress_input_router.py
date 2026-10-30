from abc import ABC, abstractmethod

from worker.domain.context.ingress_invocation_context import IngressInvocationContext
from worker.domain.model_pass.model_pass_request import ModelPassRequest
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)
from worker.domain.plan.ingress.flow_ingress_plan import FlowIngressPlan


class IngressInputRouter(ABC):
    @abstractmethod
    async def route(
        self,
        model_pass_request: ModelPassRequest,
        ingress_context: IngressInvocationContext,
        flow_plan: FlowIngressPlan,
    ) -> list[PartitionInvocationContribution | ModelPassResultContribution]: ...
