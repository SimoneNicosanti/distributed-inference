import gurobipy as gp

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.memory.memory_variables import (
    MemoryVariables,
)
from optimizer.adapters.outbound.gurobi.gurobi_units import (
    bytes_to_megabytes,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)

from .constraints import add_memory_constraints


class CompactMemoryBuilder:
    def create_variables(
        self, model: gp.Model, data: MisocpCompactFormulation
    ) -> MemoryVariables:
        variables = _add_compact_memory_variables(
            model, data, max_components=data.problem.policy.max_components
        )
        return variables

    def add_constraints(
        self,
        model: gp.Model,
        data: MisocpCompactFormulation,
        *,
        assignment: AssignmentVariables,
        memory: MemoryVariables,
    ) -> None:
        add_memory_constraints(model, data, assignment=assignment, memory=memory)


def _add_compact_memory_variables(
    model: gp.Model,
    problem: MisocpCompactFormulation,
    *,
    max_components: int,
) -> MemoryVariables:
    weights = {}
    for flow in problem.flows:
        for variant in problem.configurations_by_flow[flow]:
            for component in range(1, max_components + 1):
                for worker in problem.problem.flows[flow].workers:
                    weights[flow, variant, component, worker] = model.addVar(
                        lb=0,
                        ub=bytes_to_megabytes(
                            problem.problem.workers[
                                worker
                            ].available_inference_memory_bytes
                        ),
                        name=f"mem_weight[{problem.flow_index[flow]},{problem.configuration_index[flow, variant]},{component},{problem.worker_index[worker]}]",
                    )
    return MemoryVariables(
        component_weight_mb=weights,
        max_activation_mb={
            worker: model.addVar(
                lb=0,
                ub=bytes_to_megabytes(
                    problem.problem.workers[worker].available_inference_memory_bytes
                ),
                name=f"max_activation[{problem.worker_index[worker]}]",
            )
            for worker in problem.workers
        },
    )
