from dataclasses import dataclass

from front_end.domain.tensor_map import TensorMap


@dataclass(frozen=True, slots=True)
class InferenceResult:
    bundle: TensorMap
