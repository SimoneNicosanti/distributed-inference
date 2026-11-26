"""FCFS reference: device ingress, complete model on edge GPU, device sink.

Compute waiting is the production deterministic-service M/G/1 expression.
Network response uses the same upper utilization bins and source-interface
processor-sharing delay as the production compact formulation. Native fixed
deployment solves supply a separate feasibility certificate.
"""

import math
from typing import Any

from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    VariantData,
)
from optimizer.domain.problem.other.backend import Backend
from shared.service.service import WorkerId


def reference_workers(problem: CompactOptimizationProblem) -> tuple[WorkerId, WorkerId]:
    device = next(
        w for w, data in problem.workers.items() if data.backend == Backend.CPU
    )
    edge = next(w for w, data in problem.workers.items() if data.backend == Backend.GPU)
    return device, edge


def reference_component(variant: VariantData, layer: str) -> int:
    return (
        1
        if layer in variant.input_layers
        else 3
        if layer in variant.output_layers
        else 2
    )


def reference_estimate(
    problem: CompactOptimizationProblem, relative_error: float
) -> dict[str, Any]:
    device, edge = reference_workers(problem)
    formulation = MisocpCompactFormulationBuilder().build_formulation(
        problem, MisocpCompactFormulationConfig(relative_error=relative_error)
    )
    services = {}
    transmissions = {}
    memory = 0
    activation = 0
    for flow, data in problem.flows.items():
        key = data.configurations[0]
        variant = problem.variants[key.model_variant_id]
        services[flow] = sum(problem.execution_times_s[key, edge].values())
        memory += sum(variant.weight_bytes_by_layer.values())
        activation = max(activation, problem.configurations[key].max_cut_size_bytes)
        for cut, source, target in ((1, device, edge), (2, edge, device)):
            tensors = [
                tensor
                for tensor, route in formulation.tensor_routes_by_variant[
                    key.model_variant_id
                ].items()
                if reference_component(variant, route.source) <= cut
                and any(
                    reference_component(variant, destination) > cut
                    for destination in route.destinations
                )
            ]
            size = sum(
                problem.configurations[key].tensor_bytes[tensor] for tensor in tensors
            )
            transmissions[flow, source] = (
                size
                / formulation.connections_by_pair[
                    source, target
                ].throughput_bytes_per_sec
            )
    compute_load = sum(
        data.arrival_rate_per_s * services[flow] for flow, data in problem.flows.items()
    )
    if not math.isfinite(compute_load) or not 0 < compute_load < 1:
        raise ValueError(
            "Reference compute queue is unstable; rates are never repaired"
        )
    wait = sum(
        data.arrival_rate_per_s * services[flow] ** 2
        for flow, data in problem.flows.items()
    ) / (2 * (1 - compute_load))
    loads = {
        worker: sum(
            data.arrival_rate_per_s * transmissions[flow, worker]
            for flow, data in problem.flows.items()
        )
        for worker in (device, edge)
    }
    multipliers = {}
    for worker, load in loads.items():
        upper = next(
            (
                bound
                for bound in formulation.network_bin_limits[1:]
                if load <= bound + 1e-12
            ),
            None,
        )
        # Preserve a finite diagnostic for invalid reference networks; mark checks false.
        multipliers[worker] = (
            1 / (1 - upper) if upper is not None else 1 / (1 - min(load, 0.999999))
        )
    propagation = sum(
        formulation.connections_by_pair[pair].round_trip_time_s / 2
        for pair in ((device, edge), (edge, device))
    )
    responses = {
        str(flow.flow_id): services[flow]
        + wait
        + propagation
        + sum(
            transmissions[flow, worker] * multipliers[worker]
            for worker in (device, edge)
        )
        for flow in problem.flows
    }
    checks = {
        "compute_capacity": compute_load
        <= problem.policy.compute_utilization_limit + 1e-12,
        "network_capacity": all(
            load <= problem.policy.network_utilization_limit + 1e-12
            for load in loads.values()
        ),
        "edge_memory": memory + activation
        <= problem.workers[edge].available_inference_memory_bytes,
        "device_memory": activation
        <= problem.workers[device].available_inference_memory_bytes,
        "component_budget": problem.policy.max_components >= 3,
        "deadlines": all(
            responses[str(flow.flow_id)] <= data.max_response_time_s + 1e-12
            for flow, data in problem.flows.items()
        ),
    }
    return {
        "deployment": "device inputs -> full model on shared edge GPU -> device outputs; all active components priority=1",
        "compute_utilization": {str(device.id): 0.0, str(edge.id): compute_load},
        "offered_reference_load": compute_load,
        "network_interface_utilization": {
            str(worker.id): load for worker, load in loads.items()
        },
        "edge_memory_bytes": memory + activation,
        "device_memory_bytes": activation,
        "compute_wait_s": wait,
        "response_times_s": responses,
        "resource_and_qos_checks": checks,
        "analytic_checks_pass": all(checks.values()),
        "feasibility_certificate": False,
    }
