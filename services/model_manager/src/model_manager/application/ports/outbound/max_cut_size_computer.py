from abc import ABC, abstractmethod

from model_manager.domain.model_variant_topology import ModelVariantTopology
from model_manager.domain.shape_profile import TensorShapeProperty
from shared.model.keys import TensorKey


class MaxCutSizeComputer(ABC):
    @staticmethod
    @abstractmethod
    def compute_model_variant_max_cut_size(
        model_variant_topology: ModelVariantTopology,
        tensor_shape_properties: dict[TensorKey, TensorShapeProperty],
    ) -> float: ...
