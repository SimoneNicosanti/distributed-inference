import math
from dataclasses import dataclass

from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind,
)
from optimizer.domain.formulation.formulation_config import FormulationConfig


@dataclass(frozen=True, slots=True)
class MisocpCompactFormulationConfig(FormulationConfig):
    relative_error: float
    representation: CompactFormulationKind = CompactFormulationKind.ASSIGNMENT

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "representation", CompactFormulationKind(self.representation)
        )
        if not math.isfinite(self.relative_error) or self.relative_error <= 0:
            raise ValueError("relative_error must be finite and positive")
