import math
from dataclasses import dataclass

from optimizer.domain.formulation.formulation_config import FormulationConfig


@dataclass(frozen=True, slots=True)
class MisocpCompactFormulationConfig(FormulationConfig):
    relative_error: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.relative_error) or self.relative_error <= 0:
            raise ValueError("relative_error must be finite and positive")
