from dataclasses import dataclass

from controller.domain.optimization.ingress_sink_placement import FlowEndpointPlacement
from controller.domain.optimization.qos_constraints import QoSRequirements
from controller.domain.optimization.stream_info import StreamInfo
from shared.flow.flow import FlowId
from shared.model.model import ModelTask


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
    flow_id: FlowId
    task: ModelTask

    stream_info: StreamInfo
    qos_requirements: QoSRequirements

    ingress_sink_placement: FlowEndpointPlacement


type OptimizationRequestsByFlow = dict[FlowId, OptimizationRequest]
