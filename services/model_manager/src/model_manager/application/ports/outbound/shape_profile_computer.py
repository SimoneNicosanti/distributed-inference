from abc import ABC, abstractmethod

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.domain.input_shape_invariable_properties import (
    InputShapeInvariableTensorProperties,
)
from model_manager.domain.input_shape_variable_properties import (
    InputShapeVariableProperties,
    ModelInputConcreteShapes,
)
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import ModelVariantInfo
from model_manager.domain.model_variant_topology import ModelVariantTopology
from shared.model.keys import TensorKey


class ShapeProfileComputer(ABC):
    @abstractmethod
    def compute_shape_profiles(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
        invariable_tensor_properties: dict[
            TensorKey, InputShapeInvariableTensorProperties
        ],
    ) -> dict[ModelInputConcreteShapes, InputShapeVariableProperties]: ...
