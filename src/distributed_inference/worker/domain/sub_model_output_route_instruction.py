from pydantic import BaseModel, ConfigDict


class SubModelOutputRouteInstruction(BaseModel):
    model_config = ConfigDict(frozen=True)
    pass
