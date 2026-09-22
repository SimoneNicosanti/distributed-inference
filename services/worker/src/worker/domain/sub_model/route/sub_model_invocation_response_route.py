from pydantic import BaseModel, ConfigDict


class SubModelInvocationResponseRoute(BaseModel):
    model_config = ConfigDict(frozen=True)
    pass
