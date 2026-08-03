from abc import ABC, abstractmethod

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
)
from distributed_inference.domain.model_graph_info import ModelInfo
from distributed_inference.model_optimize.domain.optimization_level import (
    OptimizationLevel,
)


class ModelOptimizer(ABC):
    @abstractmethod
    def optimize_model(
        self,
        input_paths: MaterializedArtifact,
        output_paths: ArtifactWorkspace,
        model_info: ModelInfo,
        opt_level: OptimizationLevel,
    ) -> None: ...
