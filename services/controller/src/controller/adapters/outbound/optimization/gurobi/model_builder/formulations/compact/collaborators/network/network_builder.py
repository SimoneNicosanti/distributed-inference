import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.assignment.assignment_variables import (
    AssignmentVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators.network.network_variables import (
    NetworkVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.profile_bounds import (
    cut_upper_mb,
)
from controller.adapters.outbound.optimization.gurobi.gurobi_units import (
    bytes_per_second_to_megabytes_per_second,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from shared.flow.flow import FlowId
from shared.model.keys import TensorKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId

from .constraints import add_network_constraints


class CompactNetworkBuilder:
    def create_variables(
        self, model: gp.Model, data: CompactFormulationData
    ) -> NetworkVariables:
        variables = _add_network_variables(
            model,
            data,
            max_components=data.max_components,
            network_utilization_limit=data.network_utilization_limit,
        )
        for placement in variables.adjacent_component_workers.values():
            placement.VType = GRB.CONTINUOUS
            placement.LB = 0
            placement.UB = 1
        return variables

    def add_constraints(
        self,
        model: gp.Model,
        data: CompactFormulationData,
        *,
        assignment: AssignmentVariables,
        network: NetworkVariables,
    ) -> None:
        add_network_constraints(model, data, assignment=assignment, network=network)


def _add_network_variables(
    model: gp.Model,
    problem: CompactFormulationData,
    *,
    max_components: int,
    network_utilization_limit: float,
) -> NetworkVariables:
    tensor_crossing_auxiliary: dict[
        tuple[FlowId, ModelVariantId, TensorKey, int], gp.Var
    ] = {}
    adjacent_component_workers: dict[
        tuple[FlowId, int, WorkerId, WorkerId], gp.Var
    ] = {}
    transmission_time_s: dict[tuple[FlowId, ModelVariantId, int, WorkerId], gp.Var] = {}
    component_transmission_response_s: dict[tuple[FlowId, int], gp.Var] = {}
    cuts = range(1, max_components)
    topology = problem.network_topology

    for flow in problem.flows:
        flow_index = problem.flow_index[flow]
        for model_variant in problem.models_by_flow[flow]:
            model_index = problem.model_index[flow, model_variant]
            static_profile = problem.static_profiles_by_variant[model_variant]
            for tensor, route in static_profile.topology.tensor_routes().items():
                if len(route.maximal_destinations) <= 1:
                    continue
                tensor_index = problem.tensor_index[model_variant, tensor]
                for cut in cuts:
                    tensor_crossing_auxiliary[flow, model_variant, tensor, cut] = (
                        model.addVar(
                            lb=0.0,
                            ub=1.0,
                            name=f"c[{flow_index},{model_index},{tensor_index},{cut}]",
                        )
                    )
            for cut in cuts:
                for worker in problem.workers_by_flow[flow]:
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
            for source in problem.workers_by_flow[flow]:
                source_index = problem.worker_index[source]
                successors = topology.successors(source)
                for target in problem.workers_by_flow[flow]:
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
    connections = problem.network_topology.connections()
    for flow in problem.flows:
        for worker in problem.workers_by_flow[flow]:
            destinations = tuple(
                target
                for target in problem.workers_by_flow[flow]
                if target != worker
                and target in problem.network_topology.successors(worker)
            )
            upper[flow, worker] = max(
                (
                    cut_upper_mb(problem.static_profiles_by_variant[variant])
                    / bytes_per_second_to_megabytes_per_second(
                        connections[worker, target].throughput_bytes_per_sec
                    )
                    for variant in problem.models_by_flow[flow]
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
