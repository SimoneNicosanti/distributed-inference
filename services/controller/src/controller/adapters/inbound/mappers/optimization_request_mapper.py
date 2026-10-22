from controller.domain.optimization.optimization_request import OptimizationRequest
from controller.domain.optimization.qos_constraints import QoSRequirements
from controller.domain.optimization.stream_info import StreamInfo
from integration.dto.optimization.optimization_dto import OptimizationRequestDto


class OptimizationRequestMapper:
    @staticmethod
    def to_optimization_request(
        dto: OptimizationRequestDto,
    ) -> OptimizationRequest:
        return OptimizationRequest(
            flow_id=dto.flow_id,
            task=dto.task,
            stream_info=StreamInfo(arrival_rate=dto.stream_info.arrival_rate),
            qos_requirements=QoSRequirements(
                max_mean_response_time=(dto.qos_requirements.max_mean_response_time),
                min_model_performance=(dto.qos_requirements.min_model_performance),
                max_energy_consumption=(dto.qos_requirements.max_energy_consumption),
            ),
            ingress_id=dto.ingress_id,
        )
