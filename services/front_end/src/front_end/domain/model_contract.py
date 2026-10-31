from dataclasses import dataclass

from front_end.domain.model_specs import ModelSpecs
from shared.plan.plan_version import PlanVersion


@dataclass(frozen=True, slots=True)
class ModelContract:
    model_specs: ModelSpecs
    plan_version: PlanVersion
