from enum import StrEnum, auto
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from shared.context.model_invocation_context import (
    ModelInvocationContext,
)
from shared.plan.plan_version import PlanVersion

type ModelPassId = UUID


class ModelPassType(StrEnum):
    FORWARD = auto()


## A model invocation might be the composition of multiple model passes
## e.g. LLM inference is made up of one initial step and multiple decode steps
## We use this to distinguish between model passes
class ModelPassContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_invocation_context: ModelInvocationContext

    plan_version: PlanVersion
    model_pass_type: ModelPassType
    model_pass_id: ModelPassId = Field(default_factory=uuid4)
