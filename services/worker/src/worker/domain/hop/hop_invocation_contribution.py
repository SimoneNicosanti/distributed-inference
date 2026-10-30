from dataclasses import dataclass

from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from worker.domain.context.hop_invocation_context import (
    HopInvocationContext,
)
from worker.domain.context.model_pass_context import (
    ModelPassContext,
)
from worker.domain.hop.hop import DestinationHop, SourceHop
from worker.domain.tensor.tensor_bundle import TensorBundle


## This is the input to the execution of an hop as sent by a source
@dataclass(frozen=True, slots=True)
class HopInvocationContribution:
    source_context: HopInvocationContext[SourceHop]
    destination_hop: DestinationHop
    bundle: TensorBundle

    @property
    def plan_version(self) -> PlanVersion:
        return self.source_context.model_pass_context.plan_version

    @property
    def model_pass_context(self) -> ModelPassContext:
        return self.source_context.model_pass_context

    @property
    def source_hop(self) -> SourceHop:
        return self.source_context.hop

    @property
    def flow_id(self) -> FlowId:
        return self.model_pass_context.model_invocation_context.flow_id


## Acknowledges receipt of an hop invocation contribution.
## We ack the creation of a new invocation for the destination
@dataclass(frozen=True, slots=True)
class HopInvocationContributionAck:
    source_context: HopInvocationContext[SourceHop]
    destination_hop: DestinationHop
