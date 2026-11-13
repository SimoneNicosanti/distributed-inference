import math
from typing import override

from optimizer.application.abc.formulation_builder import FormulationBuilder
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
    TensorRoute,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
    ModelConfigurationKey,
    VariantData,
)
from shared.model.keys import LayerKey, TensorKey


class MisocpCompactFormulationBuildError(ValueError):
    pass


def _configuration_key(
    configuration: ModelConfigurationKey,
) -> tuple[str, tuple[tuple[TensorKey, tuple[int, ...]], ...]]:
    return (
        str(configuration.model_variant_id),
        tuple(sorted(configuration.input_shapes.input_shapes.items())),
    )


def _bin_limits(utilization_limit: float, relative_error: float) -> tuple[float, ...]:
    interval_count = math.ceil(
        -math.log1p(-utilization_limit) / math.log1p(relative_error)
    )
    return (
        0.0,
        *(
            1 - math.exp(-index * math.log1p(relative_error))
            for index in range(1, interval_count)
        ),
        utilization_limit,
    )


def _tensor_routes(variant: VariantData) -> dict[TensorKey, TensorRoute]:
    positions = {layer: index for index, layer in enumerate(variant.layers)}
    if len(positions) != len(variant.layers):
        raise MisocpCompactFormulationBuildError("Duplicate layers")
    if not variant.input_layers or not variant.output_layers:
        raise MisocpCompactFormulationBuildError(
            "Variant needs input and output layers"
        )
    if not set(variant.input_layers + variant.output_layers) <= positions.keys():
        raise MisocpCompactFormulationBuildError("Unknown boundary layers")
    children: dict[LayerKey, set[LayerKey]] = {layer: set() for layer in variant.layers}
    sources: dict[TensorKey, LayerKey] = {}
    destinations: dict[TensorKey, set[LayerKey]] = {}
    edges: set[tuple[LayerKey, LayerKey]] = set()
    for edge in variant.edges:
        pair = edge.source, edge.target
        if (
            edge.source not in positions
            or edge.target not in positions
            or positions[edge.source] >= positions[edge.target]
            or pair in edges
        ):
            raise MisocpCompactFormulationBuildError("Invalid topological edges")
        edges.add(pair)
        children[edge.source].add(edge.target)
        for tensor in edge.tensors:
            if tensor in sources and sources[tensor] != edge.source:
                raise MisocpCompactFormulationBuildError("Tensor has multiple sources")
            sources[tensor] = edge.source
            destinations.setdefault(tensor, set()).add(edge.target)
    descendants: dict[LayerKey, set[LayerKey]] = {}
    for layer in reversed(variant.layers):
        descendants[layer] = set(children[layer])
        for child in children[layer]:
            descendants[layer].update(descendants[child])
    return {
        tensor: TensorRoute(
            source=sources[tensor],
            destinations=tuple(sorted(targets)),
            maximal_destinations=tuple(
                target
                for target in sorted(targets)
                if not descendants[target].intersection(targets)
            ),
        )
        for tensor, targets in sorted(destinations.items())
    }


class MisocpCompactFormulationBuilder(
    FormulationBuilder[
        CompactOptimizationProblem,
        MisocpCompactFormulation,
        MisocpCompactFormulationConfig,
    ]
):
    @override
    def build_formulation(
        self,
        problem: CompactOptimizationProblem,
        formulation_config: MisocpCompactFormulationConfig,
    ) -> MisocpCompactFormulation:
        if not problem.flows or not problem.workers:
            raise MisocpCompactFormulationBuildError(
                "Flows and workers must be nonempty"
            )
        flows = tuple(sorted(problem.flows, key=str))
        workers = tuple(sorted(problem.workers, key=str))
        candidates = {}
        rates = {}
        fills = {}
        selected_variants = set()
        for flow in flows:
            data = problem.flows[flow]
            if (
                not math.isfinite(data.arrival_rate_per_s)
                or data.arrival_rate_per_s <= 0
            ):
                raise MisocpCompactFormulationBuildError(
                    "Arrival rate must be finite and positive"
                )
            if not data.workers or not set(data.workers) <= problem.workers.keys():
                raise MisocpCompactFormulationBuildError(
                    "Unknown or missing flow workers"
                )
            if not data.configurations or len(set(data.configurations)) != len(
                data.configurations
            ):
                raise MisocpCompactFormulationBuildError(
                    "Missing or duplicate configurations"
                )
            candidates[flow] = tuple(
                sorted(data.configurations, key=_configuration_key)
            )
            for configuration in candidates[flow]:
                if (
                    configuration not in problem.configurations
                    or configuration.model_variant_id not in problem.variants
                ):
                    raise MisocpCompactFormulationBuildError(
                        "Unknown configuration or variant"
                    )
                batch = problem.configurations[configuration].batch_size
                if type(batch) is not int or batch < 1:
                    raise MisocpCompactFormulationBuildError(
                        "Batch size must be a positive integer"
                    )
                rates[flow, configuration] = data.arrival_rate_per_s / batch
                fills[flow, configuration] = (batch - 1) / (2 * data.arrival_rate_per_s)
                selected_variants.add(configuration.model_variant_id)
        variants = tuple(sorted(selected_variants, key=str))
        routes = {
            variant: _tensor_routes(problem.variants[variant]) for variant in variants
        }
        for configuration in {key for values in candidates.values() for key in values}:
            variant_id = configuration.model_variant_id
            if (
                not routes[variant_id].keys()
                <= problem.configurations[configuration].tensor_bytes.keys()
            ):
                raise MisocpCompactFormulationBuildError("Missing tensor sizes")
        for (configuration, worker), times in problem.execution_times_s.items():
            if (
                configuration not in problem.configurations
                or worker not in problem.workers
            ):
                raise MisocpCompactFormulationBuildError(
                    "Unknown execution configuration or worker"
                )
            variant = problem.variants[configuration.model_variant_id]
            if not times.keys() <= set(variant.layers):
                raise MisocpCompactFormulationBuildError("Unknown execution layers")
            if any(not math.isfinite(time) or time < 0 for time in times.values()):
                raise MisocpCompactFormulationBuildError(
                    "Execution times must be finite and non-negative"
                )
        waits = {}
        limit = problem.policy.compute_utilization_limit
        for worker in workers:
            second_moment = 0.0
            maximum_time = 0.0
            for flow in flows:
                if worker not in problem.flows[flow].workers:
                    continue
                terms = []
                for configuration in candidates[flow]:
                    total = sum(
                        problem.execution_times_s.get(
                            (configuration, worker), {}
                        ).values()
                    )
                    maximum_time = max(maximum_time, total)
                    terms.append(rates[flow, configuration] * total * total)
                second_moment += max(terms)
            waits[worker] = min(second_moment, limit * maximum_time) / (
                2 * (1 - limit) ** 2
            )
        connections = {(link.source, link.target): link for link in problem.connections}
        if len(connections) != len(problem.connections) or any(
            source not in problem.workers or target not in problem.workers
            for source, target in connections
        ):
            raise MisocpCompactFormulationBuildError(
                "Duplicate or unknown network connections"
            )
        return MisocpCompactFormulation(
            problem=problem,
            flows=flows,
            workers=workers,
            configurations_by_flow=candidates,
            service_arrival_rate_per_s=rates,
            batch_fill_time_s=fills,
            compute_bin_limits=_bin_limits(limit, formulation_config.relative_error),
            network_bin_limits=_bin_limits(
                problem.policy.network_utilization_limit,
                formulation_config.relative_error,
            ),
            max_queue_wait_s_by_worker=waits,
            max_queue_wait_s_by_flow={
                flow: max(waits[worker] for worker in problem.flows[flow].workers)
                for flow in flows
            },
            flow_index={flow: index for index, flow in enumerate(flows)},
            worker_index={worker: index for index, worker in enumerate(workers)},
            configuration_index={
                (flow, configuration): index
                for flow in flows
                for index, configuration in enumerate(candidates[flow])
            },
            layer_index={
                (variant, layer): index
                for variant in variants
                for index, layer in enumerate(problem.variants[variant].layers)
            },
            tensor_index={
                (variant, tensor): index
                for variant in variants
                for index, tensor in enumerate(routes[variant])
            },
            tensor_routes_by_variant=routes,
            connections_by_pair=connections,
            successors_by_worker={
                worker: tuple(
                    target for target in workers if (worker, target) in connections
                )
                for worker in workers
            },
            predecessors_by_worker={
                worker: tuple(
                    source for source in workers if (source, worker) in connections
                )
                for worker in workers
            },
        )
