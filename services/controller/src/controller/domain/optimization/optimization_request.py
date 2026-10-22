from dataclasses import dataclass

from controller.domain.optimization.qos_constraints import QoSRequirements
from controller.domain.optimization.stream_info import StreamInfo
from shared.flow.flow import FlowId
from shared.model.model import ModelTask
from shared.service.service import WorkerId


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
    flow_id: FlowId
    task: ModelTask

    stream_info: StreamInfo
    qos_requirements: QoSRequirements

    ingress_id: WorkerId


type OptimizationRequestsByFlow = dict[FlowId, OptimizationRequest]
