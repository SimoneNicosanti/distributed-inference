import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamInfo:
    arrival_rate: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.arrival_rate) or self.arrival_rate <= 0:
            raise ValueError("Arrival rate must be finite and positive")
