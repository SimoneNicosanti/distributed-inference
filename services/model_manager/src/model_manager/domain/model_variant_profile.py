from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from model_manager.domain.layer_property import LayerProperty
from model_manager.domain.model_variant_topology import ModelVariantTopology
from model_manager.domain.shape_profile import ShapePoint, ShapeProfile
from model_manager.domain.tensor_property import TensorProperty
from shared.model.keys import LayerKey, TensorKey


@dataclass(frozen=True, slots=True)
class ModelVariantProfile:
    topology: ModelVariantTopology
    invariant_layer_properties: Mapping[
        LayerKey, LayerProperty
    ]  ## Properties that are invariant to the input shape
    invariant_tensor_properties: Mapping[
        TensorKey, TensorProperty
    ]  ## Properties that are invariant to the input shape

    shape_profiles: Mapping[
        ShapePoint, ShapeProfile
    ]  ## Properties that variant to the input shape

    optimization_contractions: tuple[
        tuple[LayerKey, ...], ...
    ]  ## Contractions done when the model is optimized for inference

    def __post_init__(self) -> None:
        for field_name in (
            "invariant_layer_properties",
            "invariant_tensor_properties",
            "shape_profiles",
        ):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )
