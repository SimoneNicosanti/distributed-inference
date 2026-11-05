from abc import ABC, abstractmethod

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.domain.input_shape_invariable_properties import (
    InputShapeInvariableLayerProperties,
    InputShapeInvariableTensorProperties,
)
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import ModelVariantInfo
from model_manager.domain.model_variant_topology import ModelVariantTopology
from shared.model.keys import LayerKey, TensorKey


class InvariantPropertiesComputer(ABC):
    @abstractmethod
    def compute_layer_invariant_properties(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
    ) -> dict[LayerKey, InputShapeInvariableLayerProperties]: ...

    @abstractmethod
    def compute_tensor_invariant_properties(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
    ) -> dict[TensorKey, InputShapeInvariableTensorProperties]: ...
