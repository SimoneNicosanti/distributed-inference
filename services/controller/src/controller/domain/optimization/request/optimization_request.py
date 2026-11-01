from dataclasses import dataclass

from controller.domain.optimization.request.flow_endpoint_placement import (
    FlowEndpointPlacement,
)
from controller.domain.optimization.request.local_worker_config import LocalWorkerConfig
from controller.domain.optimization.request.qos_constraints import QoSRequirements
from controller.domain.optimization.request.stream_info import StreamInfo
from shared.flow.flow import FlowId
from shared.model.model import ModelTask


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
    flow_id: FlowId
    task: ModelTask

    stream_info: StreamInfo
    qos_requirements: QoSRequirements

    flow_endpoint_placement: FlowEndpointPlacement
    local_worker_config: LocalWorkerConfig


type OptimizationRequestsByFlow = dict[FlowId, OptimizationRequest]
