from typing import assert_never

from controller.domain.optimization.request.flow_endpoint_placement import (
    AnyPlacement,
    FixedPlacement,
    FlowEndpointPlacement,
    Placement,
)
from controller.domain.optimization.request.local_worker_config import (
    LocalWorkerConfig,
    NoLocalWorkerConfig,
    ValidLocalWorkerConfig,
)
from controller.domain.optimization.request.model_selection_requirements import (
    ModelSelectionRequirements,
    ModelVariantSelectionStrategy,
    PerformanceSelectionStrategy,
    SelectionStrategy,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequest,
)
from controller.domain.optimization.request.qos_requirements import QoSRequirements
from controller.domain.optimization.request.stream_info import StreamInfo
from integration.dto.optimization.optimization_dto import (
    AnyPlacementDto,
    FixedPlacementDto,
    LocalWorkerConfigDto,
    ModelVariantSelectionStrategyDto,
    NoLocalWorkerConfigDto,
    OptimizationRequestDto,
    PerformanceSelectionStrategyDto,
    PlacementDto,
    SelectionStrategyDto,
    ValidLocalWorkerConfigDto,
)


class OptimizationRequestMapper:
    @staticmethod
    def to_optimization_request(
        dto: OptimizationRequestDto,
    ) -> OptimizationRequest:
        return OptimizationRequest(
            flow_id=dto.flow_id,
            model_selection_requirements=ModelSelectionRequirements(
                selection_strategy=OptimizationRequestMapper._to_selection_strategy(
                    dto.model_selection_requirements.selection_strategy
                )
            ),
            stream_info=StreamInfo(arrival_rate=dto.stream_info.arrival_rate),
            qos_requirements=QoSRequirements(
                max_mean_response_time=(dto.qos_requirements.max_mean_response_time),
                max_energy_consumption=(dto.qos_requirements.max_energy_consumption),
            ),
            flow_endpoint_placement=FlowEndpointPlacement(
                ingress_placement=OptimizationRequestMapper._to_placement(
                    dto.flow_endpoint_placement.ingress_placement
                ),
                sink_placement=OptimizationRequestMapper._to_placement(
                    dto.flow_endpoint_placement.sink_placement
                ),
            ),
            local_worker_config=OptimizationRequestMapper._to_local_worker_config(
                dto.local_worker_config
            ),
        )

    @staticmethod
    def _to_selection_strategy(dto: SelectionStrategyDto) -> SelectionStrategy:
        match dto:
            case PerformanceSelectionStrategyDto():
                return PerformanceSelectionStrategy(
                    task=dto.task,
                    min_model_performance=dto.min_model_performance,
                )
            case ModelVariantSelectionStrategyDto():
                return ModelVariantSelectionStrategy(
                    model_variant_id=dto.model_variant_id
                )
            case _:
                assert_never(dto)

    @staticmethod
    def _to_placement(dto: PlacementDto) -> Placement:
        match dto:
            case FixedPlacementDto():
                return FixedPlacement(worker_id=dto.worker_id)
            case AnyPlacementDto():
                return AnyPlacement()
            case _:
                assert_never(dto)

    @staticmethod
    def _to_local_worker_config(dto: LocalWorkerConfigDto) -> LocalWorkerConfig:
        match dto:
            case ValidLocalWorkerConfigDto():
                return ValidLocalWorkerConfig(
                    worker_id=dto.worker_id,
                    is_inference_runner=dto.is_inference_runner,
                )
            case NoLocalWorkerConfigDto():
                return NoLocalWorkerConfig()
            case _:
                assert_never(dto)
