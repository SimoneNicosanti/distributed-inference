from pydantic import BaseModel, ConfigDict


class SubModelExecutionState(BaseModel):
    model_config = ConfigDict(frozen=True)
