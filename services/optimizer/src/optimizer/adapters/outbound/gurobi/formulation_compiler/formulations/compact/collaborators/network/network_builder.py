import gurobipy as gp
from gurobipy import GRB

from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.network_variables import (
    NetworkVariables,
)
from optimizer.adapters.outbound.gurobi.gurobi_units import (
    bytes_per_second_to_megabytes_per_second,
    bytes_to_megabytes,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ModelConfigurationKey,
)
from shared.flow.flow import FlowId
from shared.model.keys import TensorKey
from shared.service.service import WorkerId

from .constraints import add_network_constraints


class CompactNetworkBuilder:
    def create_variables(
        self, model: gp.Model, data: MisocpCompactFormulation
    ) -> NetworkVariables:
        variables = _add_network_variables(
            model,
            data,
            max_components=data.problem.policy.max_components,
            network_utilization_limit=data.problem.policy.network_utilization_limit,
        )
        for placement in variables.adjacent_component_workers.values():
            placement.VType = GRB.CONTINUOUS
            placement.LB = 0
            placement.UB = 1
        return variables

    def add_constraints(
        self,
        model: gp.Model,
        data: MisocpCompactFormulation,
        *,
        assignment: AssignmentVariables,
        network: NetworkVariables,
    ) -> None:
        add_network_constraints(model, data, assignment=assignment, network=network)


def _add_network_variables(
    model: gp.Model,
    problem: MisocpCompactFormulation,
    *,
    max_components: int,
    network_utilization_limit: float,
) -> NetworkVariables:
    tensor_crossing_auxiliary: dict[
        tuple[FlowId, ModelConfigurationKey, TensorKey, int], gp.Var
    ] = {}
    adjacent_component_workers: dict[
        tuple[FlowId, int, WorkerId, WorkerId], gp.Var
    ] = {}
    transmission_time_s: dict[
        tuple[FlowId, ModelConfigurationKey, int, WorkerId], gp.Var
    ] = {}
    component_transmission_response_s: dict[tuple[FlowId, int], gp.Var] = {}
    cuts = range(1, max_components)
    topology = problem

    for flow in problem.flows:
        flow_index = problem.flow_index[flow]
        for model_variant in problem.configurations_by_flow[flow]:
            model_index = problem.configuration_index[flow, model_variant]
            for tensor, route in problem.tensor_routes_by_variant[
                model_variant.model_variant_id
            ].items():
                if len(route.maximal_destinations) <= 1:
                    continue
                tensor_index = problem.tensor_index[
                    model_variant.model_variant_id, tensor
                ]
                for cut in cuts:
                    tensor_crossing_auxiliary[flow, model_variant, tensor, cut] = (
                        model.addVar(
                            lb=0.0,
                            ub=1.0,
                            name=f"c[{flow_index},{model_index},{tensor_index},{cut}]",
                        )
                    )
            for cut in cuts:
                for worker in problem.problem.flows[flow].workers:
                    worker_index = problem.worker_index[worker]
                    transmission_time_s[flow, model_variant, cut, worker] = (
                        model.addVar(
                            lb=0.0,
                            name=f"ttx[{flow_index},{model_index},{cut},{worker_index}]",
                        )
                    )

        for cut in cuts:
            component_transmission_response_s[flow, cut] = model.addVar(
                lb=0.0,
                name=f"rtx[{flow_index},{cut}]",
            )
            for source in problem.problem.flows[flow].workers:
                source_index = problem.worker_index[source]
                successors = topology.successors_by_worker[source]
                for target in problem.problem.flows[flow].workers:
                    if source != target and target not in successors:
                        continue
                    target_index = problem.worker_index[target]
                    adjacent_component_workers[flow, cut, source, target] = (
                        model.addVar(
                            vtype=GRB.BINARY,
                            name=(
                                f"q[{flow_index},{cut},{source_index},{target_index}]"
                            ),
                        )
                    )

    network_utilization: dict[WorkerId, gp.Var] = {}
    network_bin_selected: dict[tuple[WorkerId, int], gp.Var] = {}
    network_bin_count = len(problem.network_bin_limits) - 1
    for worker in problem.workers:
        worker_index = problem.worker_index[worker]
        network_utilization[worker] = model.addVar(
            lb=0.0,
            ub=network_utilization_limit,
            name=f"rho_tx[{worker_index}]",
        )
        for bin_index in range(1, network_bin_count + 1):
            network_bin_selected[worker, bin_index] = model.addVar(
                vtype=GRB.BINARY,
                name=f"g_tx[{worker_index},{bin_index}]",
            )

    upper = {}
    responses = {}
    connections = problem.connections_by_pair
    for flow in problem.flows:
        for worker in problem.problem.flows[flow].workers:
            destinations = tuple(
                target
                for target in problem.problem.flows[flow].workers
                if target != worker and target in problem.successors_by_worker[worker]
            )
            upper[flow, worker] = max(
                (
                    bytes_to_megabytes(
                        problem.problem.configurations[variant].max_cut_size_bytes
                    )
                    / bytes_per_second_to_megabytes_per_second(
                        connections[worker, target].throughput_bytes_per_sec
                    )
                    for variant in problem.configurations_by_flow[flow]
                    for target in destinations
                ),
                default=0.0,
            )
            for cut in range(1, max_components):
                responses[flow, cut, worker] = model.addVar(
                    lb=0,
                    name=f"rtx_source[{problem.flow_index[flow]},{cut},{problem.worker_index[worker]}]",
                )
    return NetworkVariables(
        tensor_crossing_auxiliary=tensor_crossing_auxiliary,
        adjacent_component_workers=adjacent_component_workers,
        transmission_time_s=transmission_time_s,
        network_utilization=network_utilization,
        network_bin_selected=network_bin_selected,
        component_transmission_response_s=component_transmission_response_s,
        source_response_s=responses,
        nominal_upper_s=upper,
    )
