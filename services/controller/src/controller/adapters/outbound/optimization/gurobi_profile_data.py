import math
from collections.abc import Mapping

from controller.domain.optimization.optimization_context import OptimizationContext
from controller.domain.profile.model_variant_execution_profile import (
    LayerExecutionProfile,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
    ShapeProfile,
)
from integration.shared.id_json import model_variant_id_to_json
from shared.model.keys import LayerKey, TensorKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class GurobiProfileDataError(ValueError):
    pass


def fixed_shape_profile(profile: ModelVariantStaticProfile) -> ShapeProfile:
    if len(profile.shape_profiles) != 1:
        raise GurobiProfileDataError(
            f"Model variant {model_variant_id_to_json(profile.model_variant_id)} must have "
            "exactly one fixed shape profile"
        )
    return profile.shape_profiles[0]


def batch_size(profile: ModelVariantStaticProfile) -> int:
    input_points = fixed_shape_profile(profile).shape_point.input_shape_points
    if not input_points or any(not point.shape for point in input_points):
        raise GurobiProfileDataError(
            "Cannot infer the batch size of "
            f"{model_variant_id_to_json(profile.model_variant_id)}"
        )
    batch_sizes = {point.shape[0] for point in input_points}
    if len(batch_sizes) != 1:
        raise GurobiProfileDataError(
            "Input batch dimensions disagree for "
            f"{model_variant_id_to_json(profile.model_variant_id)}"
        )
    return next(iter(batch_sizes))


def tensor_size_bytes(
    profile: ModelVariantStaticProfile,
    tensor: TensorKey,
) -> int:
    invariant = profile.invariant_tensor_properties.get(tensor)
    shaped = fixed_shape_profile(profile).tensor_properties.get(tensor)
    if invariant is None or shaped is None:
        raise GurobiProfileDataError(
            f"Tensor {tensor!r} has an incomplete static profile in "
            f"{model_variant_id_to_json(profile.model_variant_id)}"
        )
    return math.prod(shaped.shape) * invariant.data_type_size_in_bytes


def layer_weight_size_bytes(
    profile: ModelVariantStaticProfile,
    layer: LayerKey,
) -> int:
    layer_property = profile.invariant_layer_properties.get(layer)
    if layer_property is None:
        return 0
    size = 0
    for weight in layer_property.weight_properties.values():
        if any(dimension < 0 for dimension in weight.shape):
            raise GurobiProfileDataError(
                f"Layer {layer!r} has a negative weight dimension in "
                f"{model_variant_id_to_json(profile.model_variant_id)}"
            )
        size += math.prod(weight.shape) * weight.data_type_size_in_bytes
    return size


def layer_activation_sizes_bytes(
    profile: ModelVariantStaticProfile,
) -> dict[LayerKey, int]:
    sizes = dict.fromkeys(profile.topology.layers(), 0)
    for tensor, route in profile.topology.tensor_routes().items():
        sizes[route.source] += tensor_size_bytes(profile, tensor)
    return sizes


def layer_execution_profiles(
    context: OptimizationContext,
    model_variant: ModelVariantId,
    worker: WorkerId,
) -> Mapping[LayerKey, LayerExecutionProfile] | None:
    worker_profiles = context.execution_profiles_by_worker.get(worker)
    if worker_profiles is None:
        return None
    execution_profile = worker_profiles.execution_profiles.get(model_variant)
    if execution_profile is None:
        return None

    shape_point = fixed_shape_profile(
        context.static_profiles_by_variant[model_variant]
    ).shape_point
    matching_profiles = tuple(
        profile
        for profile in execution_profile.shape_execution_profiles
        if profile.shape_point == shape_point
    )
    if len(matching_profiles) > 1:
        raise GurobiProfileDataError(
            f"Duplicate execution profiles for worker {worker.service_id} and model "
            f"{model_variant_id_to_json(model_variant)}"
        )
    if not matching_profiles:
        return None
    return matching_profiles[0].layer_profiles
