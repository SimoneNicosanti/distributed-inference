from pydantic import BaseModel, ConfigDict, PositiveFloat

from shared.model.model_variant import PerformanceMetric


class QoSRequirements(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_mean_response_time: PositiveFloat
    min_model_performance: PerformanceMetric
    max_energy_consumption: PositiveFloat
