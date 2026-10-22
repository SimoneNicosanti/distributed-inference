from dataclasses import dataclass

from shared.context.model_pass_context import ModelPassContext


@dataclass(frozen=True, slots=True)
class ModelPassError:
    context: ModelPassContext
    message: str
