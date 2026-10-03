from abc import ABC, abstractmethod

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import ModelVariantInfo
from model_manager.domain.model_variant_topology import ModelVariantTopology
from model_manager.domain.shape_profile import ShapePoint, ShapeProfile


class ShapeProfileComputer(ABC):
    @abstractmethod
    def compute_shape_profiles(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
    ) -> dict[ShapePoint, ShapeProfile]: ...
