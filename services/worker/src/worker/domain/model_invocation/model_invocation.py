from dataclasses import dataclass

from shared.context.model_invocation_context import ModelInvocationContext
from shared.tensor.tensor_bundle import TensorBundle


@dataclass(frozen=True, slots=True)
class ModelInvocationRequest:
    context: ModelInvocationContext
    bundle: TensorBundle


@dataclass(frozen=True, slots=True)
class ModelInvocationResult:
    context: ModelInvocationContext
    bundle: TensorBundle
