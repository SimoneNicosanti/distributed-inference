from controller.domain.backend import Backend
from controller.domain.optimization.solution.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    CutSolution,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from integration.dto.optimizer.optimization_solution_dto import OptimizationSolutionDto


class OptimizationSolutionMapper:
    @staticmethod
    def from_dto(dto: OptimizationSolutionDto) -> OptimizationSolution:
        return OptimizationSolution(
            flow_solutions=tuple(
                FlowOptimizationSolution(
                    flow_id=flow.flow_id,
                    ingress_worker_id=flow.ingress_worker_id,
                    sink_worker_id=flow.sink_worker_id,
                    model_variant_id=flow.model_variant_id,
                    components=tuple(
                        ComponentSolution(
                            component_index=component.component_index,
                            worker_id=component.worker_id,
                            backend=Backend(component.backend.value),
                            layer_keys=component.layer_keys,
                            input_layer_keys=component.input_layer_keys,
                            output_layer_keys=component.output_layer_keys,
                            tensor_contract=ComponentTensorContract(
                                component.tensor_contract.input_tensor_keys,
                                component.tensor_contract.output_tensor_keys,
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
                        CutSolution(
                            cut.source_component_index,
                            cut.destination_component_index,
                            cut.tensor_keys,
                        )
                        for cut in flow.cuts
                    ),
                    mean_batch_fill_time_s=flow.mean_batch_fill_time_s,
                    mean_response_time_s=flow.mean_response_time_s,
                    input_shapes=(
                        flow.input_shapes.input_shapes
                        if flow.input_shapes is not None
                        else None
                    ),
                )
                for flow in dto.flow_solutions
            ),
            objective_value=dto.objective_value,
            is_optimal=dto.is_optimal,
        )
