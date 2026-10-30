from dataclasses import dataclass

from worker.domain.context.model_pass_context import ModelPassContext


@dataclass(frozen=True, slots=True)
class ModelPassError:
    context: ModelPassContext
    message: str
