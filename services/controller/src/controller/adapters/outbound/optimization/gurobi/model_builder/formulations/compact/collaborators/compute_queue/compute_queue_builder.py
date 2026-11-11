import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.compute_queue.compute_queue_variables import (
    ComputeQueueVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.execution.execution_variables import (
    ExecutionVariables,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from shared.flow.flow import FlowId
from shared.service.service import WorkerId

from .constraints import add_compute_queue_constraints


class CompactComputeQueueBuilder:
    def create_variables(
        self, model: gp.Model, data: CompactFormulationData
    ) -> ComputeQueueVariables:
        variables = _add_compute_queue_variables(
            model,
            data,
            max_components=data.max_components,
            priority_count=data.priority_count,
            compute_utilization_limit=data.compute_utilization_limit,
        )
        return variables

    def add_constraints(
        self,
        model: gp.Model,
        data: CompactFormulationData,
        *,
        assignment: AssignmentVariables,
        execution: ExecutionVariables,
        compute_queue: ComputeQueueVariables,
    ) -> None:
        add_compute_queue_constraints(
            model,
            data,
            assignment=assignment,
            execution=execution,
            compute_queue=compute_queue,
        )


def _add_compute_queue_variables(
    model: gp.Model,
    problem: CompactFormulationData,
    *,
    max_components: int,
    priority_count: int,
    compute_utilization_limit: float,
) -> ComputeQueueVariables:
    component_priority: dict[tuple[FlowId, int, WorkerId, int], gp.Var] = {}
    priority_class_utilization: dict[tuple[FlowId, int, WorkerId, int], gp.Var] = {}
    component_queue_wait_s: dict[tuple[FlowId, int], gp.Var] = {}
    priorities = range(1, priority_count + 1)

    for flow in problem.flows:
        flow_index = problem.flow_index[flow]
        for component in range(1, max_components + 1):
            component_queue_wait_s[flow, component] = model.addVar(
                lb=0.0,
                ub=problem.max_queue_wait_s_by_flow[flow],
                name=f"wex[{flow_index},{component}]",
            )
            for worker in problem.workers_by_flow[flow]:
                worker_index = problem.worker_index[worker]
                for priority in priorities:
                    component_priority[flow, component, worker, priority] = (
                        model.addVar(
                            vtype=GRB.BINARY,
                            name=(
                                f"r[{flow_index},{component},{worker_index},{priority}]"
                            ),
                        )
                    )
                    priority_class_utilization[flow, component, worker, priority] = (
                        model.addVar(
                            lb=0.0,
                            ub=compute_utilization_limit,
                            name=(
                                f"rho_bar[{flow_index},{component},{worker_index},"
                                f"{priority}]"
                            ),
                        )
                    )

    priority_used: dict[tuple[WorkerId, int], gp.Var] = {}
    lower_priority_utilization: dict[tuple[WorkerId, int], gp.Var] = {}
    compute_bin_selected: dict[tuple[WorkerId, int, int], gp.Var] = {}
    queue_wait_bin_s: dict[tuple[WorkerId, int, int], gp.Var] = {}
    queue_wait_s: dict[tuple[WorkerId, int], gp.Var] = {}
    weighted_queue_wait_s: dict[tuple[WorkerId, int], gp.Var] = {}
    remaining_compute_capacity: dict[tuple[WorkerId, int], gp.Var] = {}
    compute_bin_count = len(problem.compute_bin_limits) - 1

    for worker in problem.workers:
        worker_index = problem.worker_index[worker]
        max_wait = problem.max_queue_wait_s_by_worker[worker]
        for priority in priorities:
            priority_used[worker, priority] = model.addVar(
                lb=0.0,
                ub=1.0,
                name=f"n[{worker_index},{priority}]",
            )
            queue_wait_s[worker, priority] = model.addVar(
                lb=0.0,
                ub=max_wait,
                name=f"h[{worker_index},{priority}]",
            )
            remaining_compute_capacity[worker, priority] = model.addVar(
                lb=1 - compute_utilization_limit,
                ub=1.0,
                name=f"rho_remaining[{worker_index},{priority}]",
            )
            if priority < 2:
                continue
            lower_priority_utilization[worker, priority] = model.addVar(
                lb=0.0,
                ub=compute_utilization_limit,
                name=f"rho_lt[{worker_index},{priority}]",
            )
            weighted_queue_wait_s[worker, priority] = model.addVar(
                lb=0.0,
                ub=max_wait,
                name=f"h_weighted[{worker_index},{priority}]",
            )
            for bin_index in range(1, compute_bin_count + 1):
                compute_bin_selected[worker, priority, bin_index] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"g[{worker_index},{priority},{bin_index}]",
                )
                queue_wait_bin_s[worker, priority, bin_index] = model.addVar(
                    lb=0.0,
                    ub=max_wait,
                    name=f"h_bin[{worker_index},{priority},{bin_index}]",
                )

    return ComputeQueueVariables(
        component_priority=component_priority,
        priority_used=priority_used,
        priority_class_utilization=priority_class_utilization,
        higher_priority_utilization=lower_priority_utilization,
        compute_bin_selected=compute_bin_selected,
        queue_wait_bin_s=queue_wait_bin_s,
        queue_wait_s=queue_wait_s,
        weighted_queue_wait_s=weighted_queue_wait_s,
        remaining_compute_capacity=remaining_compute_capacity,
        component_queue_wait_s=component_queue_wait_s,
    )
