from pydantic import BaseModel, ConfigDict


class RouteInstruction(BaseModel):
    model_config = ConfigDict(frozen=True)
    pass
