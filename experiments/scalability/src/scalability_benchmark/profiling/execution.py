"""Operator roofline estimates grouped by the simulated hardware backend."""

from dataclasses import dataclass

from controller.domain.backend import Backend
from controller.domain.profile.model_variant_execution_profile import (
    BackendExecutionProfile,
    LayerExecutionProfile,
    ModelVariantExecutionProfile,
    ShapeExecutionProfile,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from scalability_benchmark.configuration import ExecutionSettings, TierSettings
from scalability_benchmark.profiling.static_costs import (
    layer_weight_size_bytes,
    tensor_size_bytes,
)
from scalability_benchmark.randomness import stream


@dataclass(frozen=True)
class ExecutionEstimate:
    profile: ModelVariantExecutionProfile
    fallback_operators: tuple[str, ...]


def execution_profile(
    static: ModelVariantStaticProfile,
    hardware: TierSettings,
    settings: ExecutionSettings,
    seed: int,
    tier: str,
) -> ExecutionEstimate:
    shaped = static.shape_profiles[0]
    incoming: dict[str, set[str]] = {layer: set() for layer in static.topology.layers()}
    outgoing: dict[str, set[str]] = {layer: set() for layer in incoming}
    for tensor, route in static.topology.tensor_routes().items():
        outgoing[route.source].add(tensor)
        for destination in route.destinations:
            incoming[destination].add(tensor)
    by_operator = {
        operator: group
        for group in settings.groups.values()
        for operator in group.operators
    }
    profiles = {}
    fallback = set()
    for layer, description in static.topology.layers().items():
        if description.is_input or description.is_output:
            profiles[layer] = LayerExecutionProfile(0.0)
            continue
        efficiency = by_operator.get(description.type, settings.fallback)
        if description.type not in by_operator:
            fallback.add(description.type)
        traffic = layer_weight_size_bytes(static, layer) + sum(
            tensor_size_bytes(static, tensor)
            for tensor in incoming[layer] | outgoing[layer]
        )
        base = hardware.overhead_s + max(
            shaped.layer_properties[layer].flops
            / (hardware.compute_flops_s * efficiency.compute),
            traffic / (hardware.memory_bytes_s * efficiency.memory),
        )
        variation = stream(
            seed, "execution", static.model_variant_id.full_variant_tag, layer, tier
        ).uniform(1 - settings.jitter_fraction, 1 + settings.jitter_fraction)
        profiles[layer] = LayerExecutionProfile(base * variation, traffic)
    return ExecutionEstimate(
        ModelVariantExecutionProfile(
            (
                ShapeExecutionProfile(
                    shaped.input_shapes,
                    (BackendExecutionProfile(Backend(hardware.backend), profiles),),
                ),
            )
        ),
        tuple(sorted(fallback)),
    )
