from abc import ABC, abstractmethod

from artifacts.contracts.artifact_workspace import (
    ArtifactWorkspace,
)
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import (
    ModelVariantInfo,
)
from model_manager.domain.model_variant_topology import (
    ModelVariantTopology,
)


## NOTE: For now, we keep these calls synchronous since they are compute intensive
class ModelTopologyBuilder(ABC):
    @abstractmethod
    def build_model_topology(
        self,
        paths: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
    ) -> ModelVariantTopology: ...
