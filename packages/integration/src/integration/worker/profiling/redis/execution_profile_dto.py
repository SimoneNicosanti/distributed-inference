from pydantic import BaseModel, ConfigDict

from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class BackendLayerExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_time: float
    memory: float


class LayerExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    layer_key: LayerKey
    cpu_execution_profile: BackendLayerExecutionProfileDto
    gpu_execution_profile: BackendLayerExecutionProfileDto | None


class ShapePointDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_shape_points: dict[str, tuple[int, ...]]


class ShapeExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape_point: ShapePointDto
    layer_profiles: dict[LayerKey, LayerExecutionProfileDto]


class ModelExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_id: WorkerId
    model_variant_id: ModelVariantId
    shape_execution_profiles: tuple[ShapeExecutionProfileDto, ...]
