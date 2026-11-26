"""Reconstruct compact feasibility and costs without any optimizer API."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from optimizer.application.evaluation.queueing import QueueingMode, bin_upper
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    ModelConfigurationKey,
    ObjectiveKind,
)
from optimizer.domain.problem.other.flow_endpoint_placement import FixedPlacement
from optimizer.domain.problem.other.local_worker_config import ValidLocalWorkerConfig
from optimizer.domain.solution.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    CutSolution,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from shared.flow.flow import FlowId
from shared.model.keys import LayerKey
from shared.model.model_variant import PerformanceMetricType
from shared.service.service import WorkerId

HARD_DEADLINE_OBJECTIVES = frozenset(
    (
        ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME,
        ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_TIME,
        ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY,
    )
)


@dataclass(frozen=True, slots=True)
class FlowDecisions:
    configuration: ModelConfigurationKey
    component_by_layer: Mapping[LayerKey, int]
    workers: tuple[WorkerId, ...]
    priorities: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "component_by_layer", MappingProxyType(dict(self.component_by_layer))
        )


type Decisions = Mapping[FlowId, FlowDecisions]


@dataclass(frozen=True, slots=True)
class Evaluation:
    solution: OptimizationSolution | None
    violations: Mapping[str, float]
    compute_utilization: Mapping[WorkerId, float]
    network_utilization: Mapping[WorkerId, float]
    memory_bytes: Mapping[WorkerId, float]
    queue_wait_s: Mapping[tuple[WorkerId, int], float]
    tolerance: float

    @property
    def max_violation(self) -> float:
        return max(self.violations.values(), default=0.0)

    @property
    def valid(self) -> bool:
        return self.solution is not None and self.max_violation <= self.tolerance


def accuracy_value(
    data: MisocpCompactFormulation, flow: FlowId, configuration: ModelConfigurationKey
) -> float:
    requirement = data.problem.flows[flow].performance_requirement
    if requirement is None:
        return 0.0
    if requirement.type not in (
        PerformanceMetricType.ACCURACY,
        PerformanceMetricType.MAP_AT_50,
        PerformanceMetricType.MAP_AT_75,
    ):
        raise ValueError("Accuracy objective requires an accuracy or mAP requirement")
    values = [
        m.value
        for m in data.problem.variants[configuration.model_variant_id].accuracies
        if m.type == requirement.type
        and m.dataset == requirement.dataset
        and math.isfinite(m.value)
    ]
    if not values:
        raise ValueError("Missing matching accuracy")
    return max(values)


def objective_value(
    data: MisocpCompactFormulation,
    decisions: Decisions,
    responses: Mapping[FlowId, float],
) -> float:
    kind = data.problem.policy.objective
    rates = {f: data.problem.flows[f].arrival_rate_per_s for f in data.flows}
    total = sum(rates.values())
    if kind is ObjectiveKind.CONSTANT:
        return 0.0
    if kind is ObjectiveKind.DEADLINE_VIOLATIONS:
        return float(
            sum(
                responses[f] > data.problem.flows[f].max_response_time_s
                for f in data.flows
            )
        )
    if kind is ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY:
        return (
            sum(
                rates[f] * accuracy_value(data, f, decisions[f].configuration)
                for f in data.flows
            )
            / total
        )
    if kind is ObjectiveKind.ARRIVAL_WEIGHTED_RESPONSE_TIME:
        return sum(rates[f] * responses[f] for f in data.flows) / total
    if kind is ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_RESPONSE_TIME:
        return (
            sum(
                rates[f] * responses[f] / data.problem.flows[f].max_response_time_s
                for f in data.flows
            )
            / total
        )
    if kind is ObjectiveKind.ARRIVAL_WEIGHTED_NORMALIZED_DELAY:
        return (
            sum(
                rates[f]
                * max(0.0, responses[f] - data.problem.flows[f].max_response_time_s)
                / data.problem.flows[f].max_response_time_s
                for f in data.flows
            )
            / total
        )
    raise AssertionError(kind)


def evaluate_decisions(
    data: MisocpCompactFormulation,
    decisions: Decisions,
    *,
    mode: QueueingMode = QueueingMode.NATIVE,
    is_optimal: bool = False,
    tolerance: float = 1e-6,
) -> Evaluation:
    """Residuals: MB for memory, seconds for time, utilization for stability."""
    mode = QueueingMode(mode)
    problem, policy = data.problem, data.problem.policy
    violations: dict[str, float] = {}

    def check(name: str, residual: float) -> None:
        if not math.isfinite(residual):
            violations[name] = 1e100
        elif residual > 0:
            violations[name] = max(violations.get(name, 0.0), residual)

    def empty() -> Evaluation:
        return Evaluation(None, violations, {}, {}, {}, {}, tolerance)

    if set(decisions) != set(data.flows):
        check("flow_set", 1.0)
        return empty()
    # Validate structure before any indexed numerical reconstruction.
    for flow, decision in decisions.items():
        key, request = decision.configuration, problem.flows[flow]
        prefix = str(flow)
        if key not in data.configurations_by_flow[flow]:
            check(prefix + ":configuration", 1.0)
            continue
        graph = problem.variants[key.model_variant_id]
        n = len(decision.workers)
        if not 1 <= n <= policy.max_components or len(decision.priorities) != n:
            check(prefix + ":component_count", 1.0)
            continue
        labels = decision.component_by_layer
        if set(labels) != set(graph.layers) or any(
            type(c) is not int or not 1 <= c <= n for c in labels.values()
        ):
            check(prefix + ":layer_assignment", 1.0)
            continue
        if set(labels.values()) != set(range(1, n + 1)):
            check(prefix + ":nonempty_prefix", 1.0)
        if any(labels[layer] != 1 for layer in graph.input_layers) or any(
            labels[layer] != n for layer in graph.output_layers
        ):
            check(prefix + ":boundary_components", 1.0)
        if any(labels[e.source] > labels[e.target] for e in graph.edges):
            check(prefix + ":precedence", 1.0)
        for k, (worker, priority) in enumerate(
            zip(decision.workers, decision.priorities, strict=True), 1
        ):
            if worker not in request.workers or worker not in problem.workers:
                check(prefix + ":worker", 1.0)
                continue
            if type(priority) is not int or not 1 <= priority <= policy.priority_count:
                check(prefix + ":priority", 1.0)
            local = request.local_worker_config
            profile = problem.execution_times_s.get((key, worker), {})
            for layer in graph.layers:
                if labels[layer] != k:
                    continue
                if layer not in profile:
                    check(prefix + ":unsupported_profile", 1.0)
                if (
                    isinstance(local, ValidLocalWorkerConfig)
                    and not local.is_inference_runner
                    and local.worker_id == worker
                    and layer not in (*graph.input_layers, *graph.output_layers)
                ):
                    check(prefix + ":relay_only", 1.0)
        endpoints = request.endpoint_placement
        for rule, worker in (
            (endpoints.ingress_placement, decision.workers[0]),
            (endpoints.sink_placement, decision.workers[-1]),
        ):
            if isinstance(rule, FixedPlacement) and worker != rule.worker_id:
                check(prefix + ":endpoint", 1.0)
        for source, target in zip(
            decision.workers[:-1], decision.workers[1:], strict=True
        ):
            if source == target or (source, target) not in data.connections_by_pair:
                check(prefix + ":transition", 1.0)
        if not set(graph.layers) - set(graph.input_layers) - set(graph.output_layers):
            check(prefix + ":executable_layers", 1.0)
    if violations:
        return empty()

    rho = dict.fromkeys(data.workers, 0.0)
    rho_tx = dict.fromkeys(data.workers, 0.0)
    weights = dict.fromkeys(data.workers, 0.0)
    activation = dict.fromkeys(data.workers, 0.0)
    second = dict.fromkeys(data.workers, 0.0)
    class_rho = {
        (s, p): 0.0 for s in data.workers for p in range(1, policy.priority_count + 1)
    }
    used: set[tuple[WorkerId, int]] = set()
    services: dict[tuple[FlowId, int], float] = {}
    transmits: dict[tuple[FlowId, int], float] = {}
    latencies: dict[tuple[FlowId, int], float] = {}
    tensors: dict[tuple[FlowId, int], tuple[str, ...]] = {}
    for flow, decision in decisions.items():
        key = decision.configuration
        graph, cfg = problem.variants[key.model_variant_id], problem.configurations[key]
        rate = data.service_arrival_rate_per_s[flow, key]
        for k, (worker, priority) in enumerate(
            zip(decision.workers, decision.priorities, strict=True), 1
        ):
            assigned_layers = [
                layer
                for layer in graph.layers
                if decision.component_by_layer[layer] == k
            ]
            service = sum(
                problem.execution_times_s[key, worker][layer]
                for layer in assigned_layers
            )
            services[flow, k] = service
            rho[worker] += rate * service
            class_rho[worker, priority] += rate * service
            second[worker] += rate * service**2
            used.add((worker, priority))
            weights[worker] += sum(
                graph.weight_bytes_by_layer[layer] for layer in assigned_layers
            )
            activation[worker] = max(activation[worker], cfg.max_cut_size_bytes)
        for cut, (source, target) in enumerate(
            zip(decision.workers[:-1], decision.workers[1:], strict=True), 1
        ):
            crossing = tuple(
                t
                for t, route in data.tensor_routes_by_variant[
                    key.model_variant_id
                ].items()
                if decision.component_by_layer[route.source] <= cut
                and any(
                    decision.component_by_layer[d] > cut for d in route.destinations
                )
            )
            tensors[flow, cut] = crossing
            delta = sum(cfg.tensor_bytes[t] for t in crossing)
            link = data.connections_by_pair[source, target]
            if (
                not math.isfinite(link.throughput_bytes_per_sec)
                or link.throughput_bytes_per_sec <= 0
            ):
                check("bandwidth", 1.0)
                return empty()
            nominal = delta / link.throughput_bytes_per_sec
            transmits[flow, cut] = nominal
            latencies[flow, cut] = link.round_trip_time_s / 2
            rho_tx[source] += rate * nominal
            bandwidths = [
                c.throughput_bytes_per_sec
                for (s, t), c in data.connections_by_pair.items()
                if s == source and t in problem.flows[flow].workers and t != source
            ]
            upper = cfg.max_cut_size_bytes / min(bandwidths)
            check(f"{flow}:transmission_bound:{cut}", nominal - upper)
    waits = {}
    memory = {s: weights[s] + activation[s] for s in data.workers}
    for worker in data.workers:
        check(
            f"{worker}:compute_stability",
            rho[worker] - policy.compute_utilization_limit,
        )
        check(
            f"{worker}:network_stability",
            rho_tx[worker] - policy.network_utilization_limit,
        )
        capacity = problem.workers[worker].available_inference_memory_bytes
        check(f"{worker}:memory", (memory[worker] - capacity) / 1_000_000)
        for p in range(1, policy.priority_count + 1):
            if p > 1 and (worker, p) in used and (worker, p - 1) not in used:
                check(f"{worker}:priority_prefix:{p}", 1.0)
            higher = sum(class_rho[worker, q] for q in range(1, p))
            upto = higher + class_rho[worker, p]
            rounded = (
                bin_upper(higher, data.compute_bin_limits)
                if mode is QueueingMode.LEGACY_EQUIVALENT and p > 1
                else higher
            )
            wait = second[worker] / (
                2
                * max(1 - policy.compute_utilization_limit, 1 - rounded)
                * max(1 - policy.compute_utilization_limit, 1 - upto)
            )
            waits[worker, p] = wait
            cap = data.max_queue_wait_s_by_worker[worker]
            check(
                f"{worker}:queue_cap:{p}",
                wait - cap * (1 - policy.compute_utilization_limit if p == 1 else 1),
            )
            if mode is QueueingMode.LEGACY_EQUIVALENT and p > 1:
                check(
                    f"{worker}:queue_bin_cap:{p}",
                    wait - cap * (1 - policy.compute_utilization_limit) / (1 - rounded),
                )
    flows = []
    responses = {}
    for flow, decision in decisions.items():
        key = decision.configuration
        graph = problem.variants[key.model_variant_id]
        components = []
        for k, (worker, priority) in enumerate(
            zip(decision.workers, decision.priorities, strict=True), 1
        ):
            layers = tuple(
                layer
                for layer in graph.layers
                if decision.component_by_layer[layer] == k
            )
            nominal = transmits.get((flow, k), 0.0)
            load = rho_tx[worker]
            if mode is QueueingMode.LEGACY_EQUIVALENT:
                load = bin_upper(load, data.network_bin_limits)
            tx_response = nominal / max(1 - policy.network_utilization_limit, 1 - load)
            layer_set = set(layers)
            components.append(
                ComponentSolution(
                    component_index=k,
                    worker_id=worker,
                    backend=problem.workers[worker].backend,
                    layer_keys=layers,
                    input_layer_keys=tuple(
                        layer for layer in layers if layer in graph.input_layers
                    ),
                    output_layer_keys=tuple(
                        layer for layer in layers if layer in graph.output_layers
                    ),
                    tensor_contract=ComponentTensorContract(
                        input_tensor_keys=tuple(
                            {
                                t
                                for e in graph.edges
                                if (e.target in layer_set and e.source not in layer_set)
                                or (
                                    e.source in layer_set
                                    and e.source in graph.input_layers
                                )
                                for t in e.tensors
                            }
                        ),
                        output_tensor_keys=tuple(
                            {
                                t
                                for e in graph.edges
                                if (e.source in layer_set and e.target not in layer_set)
                                or (
                                    e.target in layer_set
                                    and e.target in graph.output_layers
                                )
                                for t in e.tensors
                            }
                        ),
                    ),
                    priority=priority,
                    mean_execution_time_s=services[flow, k],
                    mean_execution_queue_time_s=waits[worker, priority],
                    mean_transmission_time_s=tx_response,
                    transmission_latency_s=latencies.get((flow, k), 0.0),
                )
            )
        response = data.batch_fill_time_s[flow, key] + sum(
            c.mean_execution_time_s
            + c.mean_execution_queue_time_s
            + c.mean_transmission_time_s
            + c.transmission_latency_s
            for c in components
        )
        responses[flow] = response
        if policy.objective in HARD_DEADLINE_OBJECTIVES:
            check(
                f"{flow}:deadline", response - problem.flows[flow].max_response_time_s
            )
        flows.append(
            FlowOptimizationSolution(
                flow,
                decision.workers[0],
                decision.workers[-1],
                key.model_variant_id,
                tuple(components),
                tuple(
                    CutSolution(k, k + 1, tensors[flow, k])
                    for k in range(1, len(components))
                ),
                data.batch_fill_time_s[flow, key],
                response,
                key.input_shapes.input_shapes,
            )
        )
    solution = OptimizationSolution(
        tuple(flows), objective_value(data, decisions, responses), is_optimal
    )
    return Evaluation(solution, violations, rho, rho_tx, memory, waits, tolerance)


def decisions_from_solution(
    data: MisocpCompactFormulation, solution: OptimizationSolution
) -> dict[FlowId, FlowDecisions]:
    result = {}
    for flow in solution.flow_solutions:
        matches = [
            c
            for c in data.configurations_by_flow[flow.flow_id]
            if c.model_variant_id == flow.model_variant_id
            and (
                flow.input_shapes is None
                or c.input_shapes.input_shapes == flow.input_shapes
            )
        ]
        if len(matches) != 1:
            raise ValueError("Solution does not identify one eligible configuration")
        if len(
            {layer for component in flow.components for layer in component.layer_keys}
        ) != sum(len(c.layer_keys) for c in flow.components):
            raise ValueError("Layer appears in more than one component")
        if tuple(c.component_index for c in flow.components) != tuple(
            range(1, len(flow.components) + 1)
        ):
            raise ValueError("Components are not a contiguous prefix")
        result[flow.flow_id] = FlowDecisions(
            matches[0],
            {
                layer: c.component_index
                for c in flow.components
                for layer in c.layer_keys
            },
            tuple(c.worker_id for c in flow.components),
            tuple(c.priority for c in flow.components),
        )
    return result


def validate_solution(
    data: MisocpCompactFormulation,
    solution: OptimizationSolution,
    *,
    mode: QueueingMode = QueueingMode.NATIVE,
    tolerance: float = 1e-6,
    allow_epigraph_slack: bool = False,
) -> Evaluation:
    """Validate a domain solution, including its reported timing and tensor contracts.

    Legacy Gurobi outputs may retain feasible queue/response epigraph slack. When
    explicitly allowed, reported waits/response/objective may exceed their canonical
    minimum; the independent canonical objective remains available in the report.
    """
    try:
        decisions = decisions_from_solution(data, solution)
    except KeyError, ValueError:
        return Evaluation(None, {"solution_structure": 1.0}, {}, {}, {}, {}, tolerance)
    report = evaluate_decisions(
        data, decisions, mode=mode, is_optimal=solution.is_optimal, tolerance=tolerance
    )
    if report.solution is None:
        return report
    violations = dict(report.violations)

    def compare(
        name: str, reported: float, expected: float, *, lower_only: bool = False
    ) -> None:
        residual = (
            max(0.0, expected - reported) if lower_only else abs(reported - expected)
        )
        if not math.isfinite(residual):
            residual = 1e100
        if residual > 0:
            violations[name] = residual

    expected_flows = {flow.flow_id: flow for flow in report.solution.flow_solutions}
    actual_responses = {}
    for flow in solution.flow_solutions:
        expected = expected_flows[flow.flow_id]
        if (
            flow.ingress_worker_id != expected.ingress_worker_id
            or flow.sink_worker_id != expected.sink_worker_id
            or flow.cuts != expected.cuts
        ):
            violations[f"{flow.flow_id}:solution_boundaries"] = 1.0
        for actual, canonical in zip(flow.components, expected.components, strict=True):
            name = f"{flow.flow_id}:{actual.component_index}"
            if (
                actual.backend != canonical.backend
                or actual.tensor_contract != canonical.tensor_contract
                or actual.input_layer_keys != canonical.input_layer_keys
                or actual.output_layer_keys != canonical.output_layer_keys
            ):
                violations[name + ":component_contract"] = 1.0
            compare(
                name + ":service",
                actual.mean_execution_time_s,
                canonical.mean_execution_time_s,
            )
            compare(
                name + ":queue",
                actual.mean_execution_queue_time_s,
                canonical.mean_execution_queue_time_s,
                lower_only=allow_epigraph_slack,
            )
            compare(
                name + ":transmission",
                actual.mean_transmission_time_s,
                canonical.mean_transmission_time_s,
                lower_only=allow_epigraph_slack,
            )
            compare(
                name + ":latency",
                actual.transmission_latency_s,
                canonical.transmission_latency_s,
            )
        compare(
            f"{flow.flow_id}:batch_fill",
            flow.mean_batch_fill_time_s,
            expected.mean_batch_fill_time_s,
        )
        response = flow.mean_batch_fill_time_s + sum(
            c.mean_execution_time_s
            + c.mean_execution_queue_time_s
            + c.mean_transmission_time_s
            + c.transmission_latency_s
            for c in flow.components
        )
        compare(f"{flow.flow_id}:response_sum", flow.mean_response_time_s, response)
        actual_responses[flow.flow_id] = flow.mean_response_time_s
        if data.problem.policy.objective in HARD_DEADLINE_OBJECTIVES:
            excess = response - data.problem.flows[flow.flow_id].max_response_time_s
            if excess > 0:
                violations[f"{flow.flow_id}:reported_deadline"] = excess
    recomputed = objective_value(data, decisions, actual_responses)
    compare(
        "reported_objective",
        solution.objective_value,
        recomputed,
        lower_only=allow_epigraph_slack
        and data.problem.policy.objective
        not in (ObjectiveKind.CONSTANT, ObjectiveKind.ARRIVAL_WEIGHTED_ACCURACY),
    )
    return Evaluation(
        report.solution,
        violations,
        report.compute_utilization,
        report.network_utilization,
        report.memory_bytes,
        report.queue_wait_s,
        tolerance,
    )
