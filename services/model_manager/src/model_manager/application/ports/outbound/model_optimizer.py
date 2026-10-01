from abc import ABC, abstractmethod

from artifacts.workspace.artifact_workspace import (
    ArtifactWorkspace,
)
from shared.model.model import ModelInfo
from shared.model.model_version import (
    ModelVersionInfo,
)
from utils.model.optimization.api.optimization_level import (
    OptimizationLevel,
)


class ModelOptimizer(ABC):
    @abstractmethod
    def optimize_model(
        self,
        input_workspace: ArtifactWorkspace,
        output_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_version_info: ModelVersionInfo,
        opt_level: OptimizationLevel,
    ) -> None: ...
