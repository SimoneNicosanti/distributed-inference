from dataclasses import dataclass

from worker.domain.context.model_pass_context import ModelPassContext, ModelPassId
from shared.plan.plan_version import PlanVersion
from worker.domain.tensor.tensor_bundle import TensorBundle


@dataclass(frozen=True, slots=True)
class ModelPassResultContribution:
    model_pass_context: ModelPassContext
    bundle: TensorBundle

    @property
    def model_pass_id(self) -> ModelPassId:
        return self.model_pass_context.model_pass_id

    @property
    def plan_version(self) -> PlanVersion:
        return self.model_pass_context.plan_version
