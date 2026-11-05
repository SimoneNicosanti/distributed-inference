"""Transmission lower bounds; sparse transition marginals and latency are shared."""

import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi_expressions import (
    transmitted_data_size_mb_expression,
)
from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_profile_data import (
    GurobiProfileDataError,
    fixed_shape_profile,
)
from controller.adapters.outbound.optimization.gurobi_structure_constraints import (
    _add_transition_marginal_constraints,
)
from controller.adapters.outbound.optimization.gurobi_units import (
    bytes_per_second_to_megabytes_per_second,
    bytes_to_megabytes,
)
from controller.adapters.outbound.optimization.gurobi_variables import (
    AssignmentVariables,
    NetworkVariables,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)


def cut_upper_mb(profile: ModelVariantStaticProfile) -> float:
    maximum = fixed_shape_profile(profile).max_cut_size_mib
    if maximum is None:
        raise GurobiProfileDataError("compact requires profiled max_cut_size_mib")
    return bytes_to_megabytes(maximum * 1024**2)


def add_compact_transmission_constraints(
    model: gp.Model,
    problem: GurobiProblem,
    assignment: AssignmentVariables,
    network: NetworkVariables,
    *,
    context: OptimizationContext,
    requests_by_flow: OptimizationRequestsByFlow,
    max_components: int,
) -> None:
    p = problem
    connections = context.network_topology.connections()
    for flow in p.flows:
        workers = p.workers_by_flow[flow]
        flow_workers = frozenset(workers)
        max_bandwidth_mb_per_s = max(
            (
                bytes_per_second_to_megabytes_per_second(
                    connections[source, target].throughput_bytes_per_sec
                )
                for source in workers
                for target in context.network_topology.successors(source)
                if target != source and target in flow_workers
            ),
            default=0.0,
        )
        for cut in range(1, max_components):
            _add_transition_marginal_constraints(
                model,
                p,
                assignment,
                network,
                topology=context.network_topology,
                flow=flow,
                cut=cut,
            )
            # Later aggregate strengthening disabled; retain the original compact.
            # aggregate_delta = gp.LinExpr()
            # aggregate_upper = 0.0
            for variant in p.models_by_flow[flow]:
                profile = context.static_profiles_by_variant[variant]
                upper = cut_upper_mb(profile)
                delta = transmitted_data_size_mb_expression(
                    assignment,
                    network,
                    profile,
                    flow=flow,
                    model_variant=variant,
                    cut=cut,
                )
                # Later cut-volume strengthenings disabled.
                # model.addConstr(
                #     delta <= upper * assignment.component_active_at(flow, cut + 1),
                #     name=f"compact_cut_next_active[{p.flow_index[flow]},{p.model_index[flow, variant]},{cut}]",
                # )
                # model.addConstr(
                #     delta <= upper * assignment.model_selected[flow, variant],
                #     name=f"compact_cut_selected[{p.flow_index[flow]},{p.model_index[flow, variant]},{cut}]",
                # )
                # aggregate_delta += delta
                # aggregate_upper = max(aggregate_upper, upper)
                # Keep the explicitly retained transmission conservation bound.
                if max_bandwidth_mb_per_s > 0:
                    model.addConstr(
                        gp.quicksum(
                            network.transmission_time_s[flow, variant, cut, source]
                            for source in workers
                        )
                        >= delta / max_bandwidth_mb_per_s,
                        name=f"transmission_minimum[{p.flow_index[flow]},{p.model_index[flow, variant]},{cut}]",
                    )
                else:
                    # No nonlocal transition is feasible; every cut must be empty.
                    model.addConstr(
                        delta == 0,
                        name=f"transmission_no_links[{p.flow_index[flow]},{p.model_index[flow, variant]},{cut}]",
                    )
                for source in workers:
                    time = network.transmission_time_s[flow, variant, cut, source]
                    destinations = tuple(
                        target
                        for target in workers
                        if target != source
                        and target in context.network_topology.successors(source)
                    )
                    for target in destinations:
                        bandwidth = bytes_per_second_to_megabytes_per_second(
                            connections[source, target].throughput_bytes_per_sec
                        )
                        model.addConstr(
                            time
                            >= (
                                delta
                                - upper
                                * (
                                    2
                                    - assignment.component_worker[flow, cut, source]
                                    - assignment.component_worker[flow, cut + 1, target]
                                )
                            )
                            / bandwidth,
                            name=f"compact_transmission[{p.flow_index[flow]},{p.model_index[flow, variant]},{cut},{p.worker_index[source]},{p.worker_index[target]}]",
                        )
                    if not destinations:
                        time.UB = 0
            # for source in workers:
            #     aggregate_time = gp.quicksum(
            #         network.transmission_time_s[flow, variant, cut, source]
            #         for variant in p.models_by_flow[flow]
            #     )
            #     successors = context.network_topology.successors(source)
            #     for target in workers:
            #         if target == source or target not in successors:
            #             continue
            #         bandwidth = bytes_per_second_to_megabytes_per_second(
            #             connections[source, target].throughput_bytes_per_sec
            #         )
            #         model.addConstr(
            #             aggregate_time
            #             >= (
            #                 aggregate_delta
            #                 - aggregate_upper
            #                 * (
            #                     2
            #                     - assignment.component_worker[flow, cut, source]
            #                     - assignment.component_worker[flow, cut + 1, target]
            #                 )
            #             )
            #             / bandwidth,
            #             name=f"compact_transmission_aggregate[{p.flow_index[flow]},{cut},{p.worker_index[source]},{p.worker_index[target]}]",
            #         )
