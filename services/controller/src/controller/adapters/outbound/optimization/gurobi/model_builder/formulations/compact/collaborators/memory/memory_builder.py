import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.memory.memory_variables import (
    MemoryVariables,
)
from controller.adapters.outbound.optimization.gurobi.gurobi_units import (
    bytes_to_megabytes,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)

from .constraints import add_memory_constraints


class CompactMemoryBuilder:
    def create_variables(
        self, model: gp.Model, data: CompactFormulationData
    ) -> MemoryVariables:
        variables = _add_compact_memory_variables(
            model, data, max_components=data.max_components
        )
        return variables

    def add_constraints(
        self,
        model: gp.Model,
        data: CompactFormulationData,
        *,
        assignment: AssignmentVariables,
        memory: MemoryVariables,
    ) -> None:
        add_memory_constraints(model, data, assignment=assignment, memory=memory)


def _add_compact_memory_variables(
    model: gp.Model,
    problem: CompactFormulationData,
    *,
    max_components: int,
) -> MemoryVariables:
    weights = {}
    for flow in problem.flows:
        for variant in problem.models_by_flow[flow]:
            for component in range(1, max_components + 1):
                for worker in problem.workers_by_flow[flow]:
                    weights[flow, variant, component, worker] = model.addVar(
                        lb=0,
                        ub=bytes_to_megabytes(
                            problem.resource_profiles_by_worker[
                                worker
                            ].available_inference_memory_bytes
                        ),
                        name=f"mem_weight[{problem.flow_index[flow]},{problem.model_index[flow, variant]},{component},{problem.worker_index[worker]}]",
                    )
    return MemoryVariables(
        component_weight_mb=weights,
        max_activation_mb={
            worker: model.addVar(
                lb=0,
                ub=bytes_to_megabytes(
                    problem.resource_profiles_by_worker[
                        worker
                    ].available_inference_memory_bytes
                ),
                name=f"max_activation[{problem.worker_index[worker]}]",
            )
            for worker in problem.workers
        },
    )
