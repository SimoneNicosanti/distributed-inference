from pydantic import BaseModel, ConfigDict


class ServerPlan(BaseModel):
    model_config = ConfigDict(frozen=True)


class WholePlan(BaseModel):
    model_config = ConfigDict(frozen=True)
