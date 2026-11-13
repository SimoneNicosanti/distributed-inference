"""Stable Poisson streams calibrated once against the configured full pool."""

from dataclasses import dataclass

from optimizer.domain.problem.other.flow_endpoint_placement import (
    FixedPlacement,
    FlowEndpointPlacement,
)
from optimizer.domain.problem.other.local_worker_config import (
    ValidLocalWorkerConfig,
)
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.randomness import stable_uuid, stream
from shared.flow.flow import FlowId
from shared.model.model_variant import PerformanceMetric, PerformanceMetricType
from shared.service.service import WorkerId
from shared.user.user import UserId


@dataclass(frozen=True)
class SyntheticFlow:
    flow_id: FlowId
    arrival_rate_per_s: float
    max_response_time_s: float
    performance_requirement: PerformanceMetric
    max_energy_consumption: float
    endpoint_placement: FlowEndpointPlacement
    local_worker_config: ValidLocalWorkerConfig


def flow_pool(
    config: ExperimentConfig,
    seed: int,
    load: float,
    reference_time_s: float,
    device: WorkerId,
) -> tuple[SyntheticFlow, ...]:
    if reference_time_s <= 0 or not 0 < load < 1:
        raise ValueError(
            "Calibration needs positive reference service time and load in (0, 1)"
        )
    weights = [
        stream(seed, "arrival", str(index)).uniform(*config.flows.arrival_weight_range)
        for index in range(config.flows.pool_size)
    ]
    user = UserId(stable_uuid(seed, "flow-owner", "benchmark"))
    return tuple(
        SyntheticFlow(
            flow_id=FlowId(user, stable_uuid(seed, "flow", str(index))),
            arrival_rate_per_s=load * weight / (reference_time_s * sum(weights)),
            max_response_time_s=stream(seed, "deadline", str(index)).uniform(
                *config.flows.deadline_s
            ),
            performance_requirement=PerformanceMetric(
                PerformanceMetricType.ACCURACY,
                config.catalog.dataset,
                config.flows.minimum_accuracy,
            ),
            max_energy_consumption=config.flows.max_energy_consumption,
            endpoint_placement=FlowEndpointPlacement(
                FixedPlacement(device), FixedPlacement(device)
            ),
            local_worker_config=ValidLocalWorkerConfig(device, True),
        )
        for index, weight in enumerate(weights)
    )
