from dataclasses import dataclass

from shared.model.model import ModelTask
from shared.model.model_variant import ModelVariantId, PerformanceMetric


@dataclass(frozen=True, slots=True)
class PerformanceSelectionStrategy:
    task: ModelTask
    min_model_performance: PerformanceMetric


@dataclass(frozen=True, slots=True)
class ModelVariantSelectionStrategy:
    model_variant_id: ModelVariantId


type SelectionStrategy = PerformanceSelectionStrategy | ModelVariantSelectionStrategy


@dataclass(frozen=True, slots=True)
class ModelSelectionRequirements:
    selection_strategy: SelectionStrategy
