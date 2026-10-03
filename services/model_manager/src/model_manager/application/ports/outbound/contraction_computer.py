from abc import ABC, abstractmethod

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import ModelVariantInfo
from model_manager.domain.model_variant_topology import ModelVariantTopology
from shared.model.keys import LayerKey


class ContractionComputer(ABC):
    @abstractmethod
    def compute_contractions(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
    ) -> list[tuple[LayerKey, ...]]: ...
