from dataclasses import dataclass

from shared.context.model_pass_context import ModelPassContext
from shared.flow.flow import FlowId
from shared.plan.hop import PlannedInvocableHopId
from shared.plan.plan_version import PlanVersion
from worker.domain.context.hop_invocation_context import HopInvocationContext
from worker.domain.tensor.tensor_bundle import TensorBundle


## This is the input of the hop.
## It contains only the inputs of the partition, not the skip tensors.
@dataclass(frozen=True, slots=True)
class HopInvocationRequest[HopT: PlannedInvocableHopId]:
    context: HopInvocationContext[HopT]

    payload: TensorBundle

    @property
    def plan_version(self) -> PlanVersion:
        return self.context.model_pass_context.plan_version

    @property
    def flow_id(self) -> FlowId:
        return self.context.model_pass_context.model_invocation_context.flow_id

    @property
    def hop(self) -> HopT:
        return self.context.hop

    @property
    def model_pass_context(self) -> ModelPassContext:
        return self.context.model_pass_context


## This is the output of the sub-model as returned to the caller
@dataclass(frozen=True, slots=True)
class HopInvocationResult[HopT: PlannedInvocableHopId]:
    context: HopInvocationContext[HopT]

    payload: TensorBundle

    @property
    def model_pass_context(self) -> ModelPassContext:
        return self.context.model_pass_context

    @property
    def flow_id(self) -> FlowId:
        return self.model_pass_context.model_invocation_context.flow_id
