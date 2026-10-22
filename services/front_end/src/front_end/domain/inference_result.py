from dataclasses import dataclass

from shared.tensor.tensor_bundle import TensorBundle


@dataclass(frozen=True, slots=True)
class InferenceResult:
    bundle: TensorBundle
