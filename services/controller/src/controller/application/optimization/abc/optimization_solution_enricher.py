from abc import ABC, abstractmethod
from collections.abc import Mapping

from controller.domain.optimization.solution.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.optimization.solution.optimization_solution import (
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
