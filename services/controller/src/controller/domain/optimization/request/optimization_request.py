from dataclasses import dataclass

from controller.domain.optimization.request.flow_endpoint_placement import (
    FlowEndpointPlacement,
)
from controller.domain.optimization.request.qos_constraints import QoSRequirements
from controller.domain.optimization.request.stream_info import StreamInfo
from shared.flow.flow import FlowId
from shared.model.model import ModelTask
from shared.service.service import WorkerId


@dataclass(frozen=True, slots=True)
class LocalWorkerConfig:
    worker_id: WorkerId
    is_inference_runner: bool


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
    flow_id: FlowId
    task: ModelTask

    stream_info: StreamInfo
    qos_requirements: QoSRequirements

    ingress_sink_placement: FlowEndpointPlacement
    local_worker_config: LocalWorkerConfig | None


type OptimizationRequestsByFlow = dict[FlowId, OptimizationRequest]
