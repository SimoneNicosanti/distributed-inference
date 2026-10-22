import math
from dataclasses import dataclass

from shared.model.model_variant import PerformanceMetric


@dataclass(frozen=True, slots=True)
class QoSRequirements:
    max_mean_response_time: float
    min_model_performance: PerformanceMetric
    max_energy_consumption: float

    def __post_init__(self) -> None:
        values = self.max_mean_response_time, self.max_energy_consumption
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("QoS limits must be finite and positive")
