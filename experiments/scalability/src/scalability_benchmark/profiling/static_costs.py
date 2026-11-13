"""Static costs for the benchmark's single-shape synthetic profiles."""

import math

from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from shared.model.keys import LayerKey, TensorKey


def tensor_size_bytes(profile: ModelVariantStaticProfile, tensor: TensorKey) -> int:
    if len(profile.shape_profiles) != 1:
        raise ValueError("Synthetic profiles must have exactly one shape configuration")
    invariant = profile.invariant_tensor_properties[tensor]
    shaped = profile.shape_profiles[0].tensor_properties[tensor]
    return math.prod(shaped.shape) * invariant.data_type_size_in_bytes


def layer_weight_size_bytes(profile: ModelVariantStaticProfile, layer: LayerKey) -> int:
    properties = profile.invariant_layer_properties.get(layer)
    if properties is None:
        return 0
    total = 0
    for weight in properties.weight_properties.values():
        if any(dimension < 0 for dimension in weight.shape):
            raise ValueError("Weight dimensions must be non-negative")
        total += math.prod(weight.shape) * weight.data_type_size_in_bytes
    return total
