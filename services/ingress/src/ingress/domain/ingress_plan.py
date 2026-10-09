from pydantic import BaseModel, ConfigDict


class IngressPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_bindings: list[str]
    output_bindings: list[str]
