import math
from dataclasses import dataclass

from shared.service.service import WorkerId


@dataclass(frozen=True, slots=True)
class NetworkConnection:
    source: WorkerId
    target: WorkerId
    throughput_bytes_per_sec: float
    round_trip_time_s: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.throughput_bytes_per_sec)
            or self.throughput_bytes_per_sec <= 0
        ):
            raise ValueError("Throughput must be finite and positive")
        if not math.isfinite(self.round_trip_time_s) or self.round_trip_time_s < 0:
            raise ValueError("Round-trip time must be finite and non-negative")
