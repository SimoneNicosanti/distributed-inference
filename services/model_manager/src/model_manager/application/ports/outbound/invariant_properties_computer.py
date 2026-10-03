from abc import ABC, abstractmethod

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.domain.layer_property import LayerProperty
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import ModelVariantInfo
from model_manager.domain.model_variant_topology import ModelVariantTopology
from model_manager.domain.tensor_property import TensorProperty
from shared.model.keys import LayerKey, TensorKey


class InvariantPropertiesComputer(ABC):
    @abstractmethod
    def compute_layer_invariant_properties(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
    ) -> dict[LayerKey, LayerProperty]: ...

    @abstractmethod
    def compute_tensor_invariant_properties(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
    ) -> dict[TensorKey, TensorProperty]: ...
