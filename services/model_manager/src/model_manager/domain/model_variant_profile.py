from pydantic import BaseModel, ConfigDict

from model_manager.domain.layer_property import LayerProperty
from model_manager.domain.model_variant_topology import ModelVariantTopology
from model_manager.domain.shape_profile import ShapePoint, ShapeProfile
from model_manager.domain.tensor_property import TensorProperty
from shared.model.keys import LayerKey, TensorKey


class ModelVariantProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    topology: ModelVariantTopology
    invariant_layer_properties: dict[
        LayerKey, LayerProperty
    ]  ## Properties that are invariant to the input shape
    invariant_tensor_properties: dict[
        TensorKey, TensorProperty
    ]  ## Properties that are invariant to the input shape

    shape_profiles: dict[ShapePoint, ShapeProfile]

    optimization_contractions: list[
        tuple[LayerKey, ...]
    ]  ## Contractions done when the model is optimized for inference
