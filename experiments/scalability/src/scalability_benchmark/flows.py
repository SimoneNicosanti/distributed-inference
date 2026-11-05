"""Stable Poisson streams calibrated once against the configured full pool."""

from controller.domain.optimization.request.flow_endpoint_placement import (
    FixedPlacement,
    FlowEndpointPlacement,
)
from controller.domain.optimization.request.local_worker_config import (
    ValidLocalWorkerConfig,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequest,
)
from controller.domain.optimization.request.qos_constraints import QoSRequirements
from controller.domain.optimization.request.stream_info import StreamInfo
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.randomness import stable_uuid, stream
from shared.flow.flow import FlowId
from shared.model.model import ModelTask
from shared.model.model_variant import PerformanceMetric, PerformanceMetricType
from shared.service.service import WorkerId
from shared.user.user import UserId


def flow_pool(
    config: ExperimentConfig,
    seed: int,
    load: float,
    reference_time_s: float,
    device: WorkerId,
) -> tuple[OptimizationRequest, ...]:
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
        OptimizationRequest(
            flow_id=FlowId(user, stable_uuid(seed, "flow", str(index))),
            task=ModelTask.CLASSIFICATION,
            stream_info=StreamInfo(load * weight / (reference_time_s * sum(weights))),
            qos_requirements=QoSRequirements(
                max_mean_response_time=stream(seed, "deadline", str(index)).uniform(
                    *config.flows.deadline_s
                ),
                min_model_performance=PerformanceMetric(
                    PerformanceMetricType.ACCURACY,
                    config.catalog.dataset,
                    config.flows.minimum_accuracy,
                ),
                max_energy_consumption=config.flows.max_energy_consumption,
            ),
            flow_endpoint_placement=FlowEndpointPlacement(
                FixedPlacement(device), FixedPlacement(device)
            ),
            local_worker_config=ValidLocalWorkerConfig(device, True),
        )
        for index, weight in enumerate(weights)
    )
