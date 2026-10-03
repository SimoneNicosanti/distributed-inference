from abc import ABC, abstractmethod

from artifacts.contracts.artifact_workspace import (
    ArtifactWorkspace,
)
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import (
    ModelVariantInfo,
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
        model_variant_info: ModelVariantInfo,
        opt_level: OptimizationLevel,
    ) -> None: ...
