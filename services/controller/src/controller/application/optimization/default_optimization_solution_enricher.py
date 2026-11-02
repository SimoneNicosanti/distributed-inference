from collections.abc import Mapping
from typing import override

from controller.application.optimization.abc.optimization_solution_enricher import (
    OptimizationSolutionEnricher,
)
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
    TensorProperty,
)
from shared.model.keys import TensorKey
from shared.model.model_variant import ModelVariantId


class DefaultOptimizationSolutionEnricher(OptimizationSolutionEnricher):
    @override
    def enrich_solution(
        self,
        solution: OptimizationSolution,
        static_profiles_by_variant: Mapping[ModelVariantId, ModelVariantStaticProfile],
    ) -> EnrichedOptimizationSolution:
        input_properties_by_variant: dict[
            ModelVariantId, Mapping[TensorKey, TensorProperty]
        ] = {}

        for flow in solution.flow_solutions:
            variant_id = flow.model_variant_id
            if variant_id in input_properties_by_variant:
                continue

            profile = static_profiles_by_variant.get(variant_id)
            if profile is None:
                raise ValueError(f"No static profile for selected variant {variant_id}")
            if profile.model_variant_id != variant_id:
                raise ValueError(
                    "Static profile key does not match the selected model variant"
                )

            layers = profile.topology.layers()
            input_tensor_keys = {
                tensor_key
                for edge in profile.topology.edges().values()
                if layers[edge.source].is_input
                for tensor_key in edge.tensors
            }
            if not input_tensor_keys:
                raise ValueError(
                    f"Selected variant {variant_id} has no model input tensors"
                )

            missing_tensors = (
                input_tensor_keys - profile.invariant_tensor_properties.keys()
            )
            if missing_tensors:
                raise ValueError(
                    f"Selected variant {variant_id} is missing input tensor properties: "
                    f"{sorted(missing_tensors)!r}"
                )

            input_properties_by_variant[variant_id] = {
                tensor_key: profile.invariant_tensor_properties[tensor_key]
                for tensor_key in sorted(input_tensor_keys)
            }

        return EnrichedOptimizationSolution(
            solution=solution,
            input_tensor_properties_by_variant=input_properties_by_variant,
        )
