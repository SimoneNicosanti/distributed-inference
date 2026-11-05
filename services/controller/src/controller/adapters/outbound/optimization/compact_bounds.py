"""Compact-only physical queue bounds; shared preprocessing is unchanged."""

from dataclasses import replace

from controller.adapters.outbound.optimization.gurobi_problem import GurobiProblem
from controller.adapters.outbound.optimization.gurobi_profile_data import (
    layer_execution_profiles,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)
from controller.domain.optimization.solving.optimization_context import (
    OptimizationContext,
)
from controller.domain.optimization.solving.solver_config import ProblemConfig


def prepare_compact_problem(
    problem: GurobiProblem,
    *,
    context: OptimizationContext,
    requests_by_flow: OptimizationRequestsByFlow,
    problem_config: ProblemConfig,
) -> GurobiProblem:
    waits = {}
    limit = problem_config.compute_utilization_limit
    for worker in problem.workers:
        second_moment = 0.0
        maximum_time = 0.0
        for flow in problem.flows:
            if worker not in problem.workers_by_flow[flow]:
                continue
            candidates = []
            for variant in problem.models_by_flow[flow]:
                budget = (
                    float(
                        requests_by_flow[flow].qos_requirements.max_mean_response_time
                    )
                    - problem.batch_fill_time_s[flow, variant]
                )
                if budget < 0:
                    continue
                profiles = layer_execution_profiles(context, variant, worker) or {}
                time = min(
                    sum(
                        float(profile.execution_time_s) for profile in profiles.values()
                    ),
                    budget,
                )
                maximum_time = max(maximum_time, time)
                candidates.append(
                    problem.service_arrival_rate_per_s[flow, variant] * time * time
                )
            second_moment += max(candidates, default=0.0)
        bound = min(second_moment, limit * maximum_time)
        waits[worker] = bound / (2 * (1 - limit) ** 2)
    return replace(
        problem,
        max_queue_wait_s_by_worker=waits,
        max_queue_wait_s_by_flow={
            flow: min(
                float(requests_by_flow[flow].qos_requirements.max_mean_response_time),
                max(waits[worker] for worker in problem.workers_by_flow[flow]),
            )
            for flow in problem.flows
        },
    )
