from abc import ABC, abstractmethod
from collections.abc import Mapping

from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from shared.model.model_variant import ModelVariantId


class OptimizationSolutionEnricher(ABC):
    @abstractmethod
    def enrich_solution(
        self,
        solution: OptimizationSolution,
        static_profiles_by_variant: Mapping[ModelVariantId, ModelVariantStaticProfile],
    ) -> EnrichedOptimizationSolution: ...
