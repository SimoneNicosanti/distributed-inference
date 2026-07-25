from abc import ABC, abstractmethod
from enum import Enum

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.domain.model_graph_info import ModelInfo


class OptimizationLevel(Enum):
    NONE = 0
    BASIC = 1
    EXTENDED = 2


class ModelOptimizer(ABC):
    @abstractmethod
    def optimize_model(
        self,
        input_paths: ArtifactConcretePaths,
        output_paths: ArtifactConcretePaths,
        model_info: ModelInfo,
        opt_level: OptimizationLevel,
    ) -> None: ...
