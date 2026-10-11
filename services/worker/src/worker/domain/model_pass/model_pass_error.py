from pydantic import BaseModel

from shared.context.model_pass_context import ModelPassContext


class ModelPassError(BaseModel):
    context: ModelPassContext
    message: str
