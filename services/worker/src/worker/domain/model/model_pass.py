from dataclasses import dataclass
from typing import TypeAlias

from shared.context.model_pass_context import ModelPassContext
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from worker.domain.tensor.tensor_bundle import TensorBundle


@dataclass(frozen=True, slots=True)
class ModelPassRequest:
    context: ModelPassContext
    bundle: TensorBundle


@dataclass(frozen=True, slots=True)
class ModelPassError:
    context: ModelPassContext
    message: str

    @property
    def flow_id(self) -> FlowId:
        return self.context.flow_id

    @property
    def plan_version(self) -> PlanVersion:
        return self.context.plan_version


@dataclass(frozen=True, slots=True)
class ModelPassResult:
    context: ModelPassContext
    bundle: TensorBundle


ModelPassOutcome: TypeAlias = ModelPassResult | ModelPassError
