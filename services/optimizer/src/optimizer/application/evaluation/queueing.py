"""Queue semantics shared by independent evaluation and nonlinear adapters."""

from enum import StrEnum


class QueueingMode(StrEnum):
    NATIVE = "native-nonlinear"
    LEGACY_EQUIVALENT = "legacy-equivalent"


def bin_upper(value: float, limits: tuple[float, ...]) -> float:
    """Smallest admissible upper endpoint, including the lower bin at ties."""
    return next((upper for upper in limits[1:] if value <= upper), limits[-1])
