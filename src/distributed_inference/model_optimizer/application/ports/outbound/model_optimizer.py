from abc import ABC, abstractmethod

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
)
from distributed_inference.model_manager.domain.model import ModelInfo
from distributed_inference.model_manager.domain.model_version import ModelVersionInfo
from distributed_inference.model_optimizer.domain.optimization_level import (
    OptimizationLevel,
)


class ModelOptimizer(ABC):
    @abstractmethod
    def optimize_model(
        self,
        input_paths: MaterializedArtifact,
        output_paths: ArtifactWorkspace,
        model_info: ModelInfo,
        model_version_info: ModelVersionInfo,
        opt_level: OptimizationLevel,
    ) -> None: ...
