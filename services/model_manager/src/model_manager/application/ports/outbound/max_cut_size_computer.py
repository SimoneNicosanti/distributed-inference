from abc import ABC, abstractmethod

from model_manager.domain.input_shape_invariable_properties import (
    InputShapeInvariableTensorProperties,
)
from model_manager.domain.input_shape_variable_properties import (
    InputShapeVariableTensorProperties,
)
from model_manager.domain.model_variant_topology import ModelVariantTopology
from shared.model.keys import TensorKey


class MaxCutSizeComputer(ABC):
    @staticmethod
    @abstractmethod
    def compute_model_variant_max_cut_size(
        model_variant_topology: ModelVariantTopology,
        invariable_tensor_properties: dict[
            TensorKey, InputShapeInvariableTensorProperties
        ],
        variable_tensor_properties: dict[TensorKey, InputShapeVariableTensorProperties],
    ) -> float:
        """Return the maximum cut tensor volume in MiB (1024 ** 2 bytes)."""
        ...
