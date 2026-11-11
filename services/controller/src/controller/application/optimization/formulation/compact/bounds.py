"""Physical queue bounds independent of response deadlines and objective."""

from dataclasses import replace

from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)


def prepare_compact_problem(problem: CompactFormulationData) -> CompactFormulationData:
    waits = {}
    limit = problem.compute_utilization_limit
    for worker in problem.workers:
        second_moment = 0.0
        maximum_time = 0.0
        for flow in problem.flows:
            if worker not in problem.workers_by_flow[flow]:
                continue
            candidates = []
            for variant in problem.models_by_flow[flow]:
                profiles = (
                    problem.layer_execution_profiles_by_model_worker[variant, worker]
                    or {}
                )
                time = sum(
                    float(profile.execution_time_s) for profile in profiles.values()
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
            flow: max(waits[worker] for worker in problem.workers_by_flow[flow])
            for flow in problem.flows
        },
    )
