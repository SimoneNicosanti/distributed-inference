from integration.dto.optimization.optimization_dto import FixedPlacementDto
from integration.dto.optimizer.common import OptimizerDto
from integration.dto.optimizer.optimization_problem_dto import (
    OptimizationProblemDto,
    configuration_identity,
)
from integration.dto.optimizer.optimization_solution_dto import OptimizationSolutionDto


class OptimizeRequest(OptimizerDto):
    problem: OptimizationProblemDto


class OptimizeResponse(OptimizerDto):
    solution: OptimizationSolutionDto


def validate_solution_for_problem(
    problem: OptimizationProblemDto, solution: OptimizationSolutionDto
) -> None:
    """Check response identities against the submitted problem, without re-solving it."""
    flows = {flow.flow_id: flow for flow in problem.flows}
    if {flow.flow_id for flow in solution.flow_solutions} != set(flows):
        raise ValueError("Optimizer returned a different set of flows")
    workers = {worker.worker_id: worker for worker in problem.workers}
    variants = {variant.model_variant_id: variant for variant in problem.variants}
    for flow in solution.flow_solutions:
        requested = flows[flow.flow_id]
        candidates = [
            key
            for key in requested.configurations
            if key.model_variant_id == flow.model_variant_id
        ]
        if not candidates:
            raise ValueError("Optimizer returned an ineligible model variant")
        if flow.input_shapes is None:
            if len(candidates) != 1:
                raise ValueError("Optimizer must identify the selected input shapes")
        else:
            selected = (
                flow.model_variant_id,
                tuple(sorted(flow.input_shapes.input_shapes.items())),
            )
            if selected not in {configuration_identity(key) for key in candidates}:
                raise ValueError("Optimizer returned an ineligible input configuration")
        if not {flow.ingress_worker_id, flow.sink_worker_id} <= set(requested.workers):
            raise ValueError("Optimizer returned an ineligible endpoint worker")
        for placement, actual in (
            (requested.endpoint_placement.ingress_placement, flow.ingress_worker_id),
            (requested.endpoint_placement.sink_placement, flow.sink_worker_id),
        ):
            if isinstance(placement, FixedPlacementDto) and str(
                placement.worker_id.id
            ) != str(actual.id):
                raise ValueError("Optimizer changed a fixed endpoint placement")
        layers = set(variants[flow.model_variant_id].layers)
        for component in flow.components:
            if component.worker_id not in requested.workers:
                raise ValueError("Optimizer returned an ineligible component worker")
            if component.backend != workers[component.worker_id].backend:
                raise ValueError("Optimizer returned a different worker backend")
            if not set(component.layer_keys) <= layers:
                raise ValueError("Optimizer returned an unknown layer")
