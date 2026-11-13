from integration.dto.model.profiled_model_variant_dto import ModelInputConcreteShapesDto
from integration.dto.optimizer.optimization_solution_dto import (
    ComponentSolutionDto,
    ComponentTensorContractDto,
    CutSolutionDto,
    FlowOptimizationSolutionDto,
    OptimizationSolutionDto,
)
from integration.dto.worker_profiles.execution_profile_dto import BackendDto
from optimizer.domain.solution.optimization_solution import OptimizationSolution


class OptimizationSolutionMapper:
    @staticmethod
    def to_dto(solution: OptimizationSolution) -> OptimizationSolutionDto:
        return OptimizationSolutionDto(
            flow_solutions=tuple(
                FlowOptimizationSolutionDto(
                    flow_id=flow.flow_id,
                    ingress_worker_id=flow.ingress_worker_id,
                    sink_worker_id=flow.sink_worker_id,
                    model_variant_id=flow.model_variant_id,
                    components=tuple(
                        ComponentSolutionDto(
                            component_index=component.component_index,
                            worker_id=component.worker_id,
                            backend=BackendDto(component.backend.value),
                            layer_keys=component.layer_keys,
                            input_layer_keys=component.input_layer_keys,
                            output_layer_keys=component.output_layer_keys,
                            tensor_contract=ComponentTensorContractDto(
                                input_tensor_keys=component.tensor_contract.input_tensor_keys,
                                output_tensor_keys=component.tensor_contract.output_tensor_keys,
                            ),
                            priority=component.priority,
                            mean_execution_time_s=component.mean_execution_time_s,
                            mean_execution_queue_time_s=component.mean_execution_queue_time_s,
                            mean_transmission_time_s=component.mean_transmission_time_s,
                            transmission_latency_s=component.transmission_latency_s,
                        )
                        for component in flow.components
                    ),
                    cuts=tuple(
                        CutSolutionDto(
                            source_component_index=cut.source_component_index,
                            destination_component_index=cut.destination_component_index,
                            tensor_keys=cut.tensor_keys,
                        )
                        for cut in flow.cuts
                    ),
                    mean_batch_fill_time_s=flow.mean_batch_fill_time_s,
                    mean_response_time_s=flow.mean_response_time_s,
                    input_shapes=(
                        ModelInputConcreteShapesDto(
                            input_shapes=dict(flow.input_shapes)
                        )
                        if flow.input_shapes is not None
                        else None
                    ),
                )
                for flow in solution.flow_solutions
            ),
            objective_value=solution.objective_value,
            is_optimal=solution.is_optimal,
        )
