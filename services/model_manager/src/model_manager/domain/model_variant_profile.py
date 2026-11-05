from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from model_manager.domain.input_shape_invariable_properties import (
    InputShapeInvariableProperties,
)
from model_manager.domain.input_shape_variable_properties import (
    InputShapeVariableProperties,
    ModelInputConcreteShapes,
)
from model_manager.domain.model_variant_topology import ModelVariantTopology
from shared.model.keys import LayerKey


@dataclass(frozen=True, slots=True)
class ModelVariantProfile:
    topology: ModelVariantTopology

    invariable_properties: InputShapeInvariableProperties

    variable_properties: Mapping[ModelInputConcreteShapes, InputShapeVariableProperties]

    optimization_contractions: tuple[
        tuple[LayerKey, ...], ...
    ]  ## Contractions done when the model is optimized for inference

    def __post_init__(self) -> None:
        for field_name in ("variable_properties",):
            object.__setattr__(
                self,
                field_name,
                MappingProxyType(dict(getattr(self, field_name))),
            )
