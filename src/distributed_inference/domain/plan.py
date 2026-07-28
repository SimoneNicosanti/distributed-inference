from pydantic import BaseModel, ConfigDict


class InferencePlanVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    version_number: int


class ServiceInferencePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: InferencePlanVersion


class WholeInferencePlan(BaseModel):
    model_config = ConfigDict(frozen=True)
