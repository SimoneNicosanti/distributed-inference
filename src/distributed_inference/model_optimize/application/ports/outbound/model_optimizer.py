from abc import ABC, abstractmethod

from distributed_inference.domain.model_graph_info import ModelInfo
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.model_optimize.domain.optimization_level import (
    OptimizationLevel,
)


class ModelOptimizer(ABC):
    @abstractmethod
    def optimize_model(
        self,
        input_paths: ArtifactConcretePaths,
        output_paths: ArtifactConcretePaths,
        model_info: ModelInfo,
        opt_level: OptimizationLevel,
    ) -> None: ...
