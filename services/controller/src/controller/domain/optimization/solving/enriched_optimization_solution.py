from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.profile.model_variant_static_profile import TensorProperty
from shared.model.keys import TensorKey
from shared.model.model_variant import ModelVariantId


@dataclass(frozen=True, slots=True)
class EnrichedOptimizationSolution:
    solution: OptimizationSolution
    input_tensor_properties_by_variant: Mapping[
        ModelVariantId, Mapping[TensorKey, TensorProperty]
    ]

    def __post_init__(self) -> None:
        flow_ids = [flow.flow_id for flow in self.solution.flow_solutions]
        if len(flow_ids) != len(set(flow_ids)):
            raise ValueError("Flow ids in an enriched solution must be unique")

        selected_variants = {
            flow.model_variant_id for flow in self.solution.flow_solutions
        }
        if set(self.input_tensor_properties_by_variant) != selected_variants:
            raise ValueError(
                "Input tensor properties must match the selected model variants"
            )

        if any(
            not properties
            for properties in self.input_tensor_properties_by_variant.values()
        ):
            raise ValueError(
                "Selected model variants must have input tensor properties"
            )

        input_properties_by_variant = {
            variant: MappingProxyType(dict(properties))
            for variant, properties in self.input_tensor_properties_by_variant.items()
        }
        object.__setattr__(
            self,
            "input_tensor_properties_by_variant",
            MappingProxyType(input_properties_by_variant),
        )
