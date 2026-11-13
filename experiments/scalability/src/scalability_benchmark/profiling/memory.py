"""Liveness diagnostics; production memory constraints remain unchanged."""

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from scalability_benchmark.profiling.static_costs import (
    tensor_size_bytes,
)
from shared.model.keys import LayerKey


@dataclass(frozen=True)
class MemoryEstimate:
    persistent_weights_bytes: int
    peak_live_activations_bytes: int
    summed_activations_bytes: int

    def as_record(self) -> dict[str, int]:
        return {
            **asdict(self),
            "weights_plus_peak_bytes": self.persistent_weights_bytes
            + self.peak_live_activations_bytes,
        }


def estimate_memory(
    profile: ModelVariantStaticProfile,
    partition_layers: Iterable[LayerKey] | None = None,
) -> MemoryEstimate:
    order = profile.topology.layers_in_topological_order()
    selected = set(order if partition_layers is None else partition_layers)
    if not selected or selected - set(order):
        raise ValueError("Memory estimate needs nonempty known partition layers")
    execution = [layer for layer in order if layer in selected]
    positions = {layer: index for index, layer in enumerate(execution)}
    weights = {}
    for layer in execution:
        for key, weight in profile.invariant_layer_properties[
            layer
        ].weight_properties.items():
            weights[key] = math.prod(weight.shape) * weight.data_type_size_in_bytes
    intervals = []
    for tensor, route in profile.topology.tensor_routes().items():
        consumers = [
            positions[layer] for layer in route.destinations if layer in selected
        ]
        if route.source not in selected and not consumers:
            continue
        birth = positions.get(route.source, 0)
        death = max(consumers, default=birth)
        if route.source in selected and any(
            layer not in selected for layer in route.destinations
        ):
            death = len(execution) - 1
        intervals.append((birth, death, tensor_size_bytes(profile, tensor)))
    peak = max(
        sum(size for birth, death, size in intervals if birth <= index <= death)
        for index in range(len(execution))
    )
    return MemoryEstimate(
        sum(weights.values()), peak, sum(size for _, _, size in intervals)
    )
