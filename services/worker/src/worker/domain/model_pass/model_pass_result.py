from dataclasses import dataclass

from worker.domain.context.model_pass_context import ModelPassContext
from worker.domain.tensor.tensor_bundle import TensorBundle


@dataclass(frozen=True, slots=True)
class ModelPassResult:
    context: ModelPassContext
    bundle: TensorBundle
