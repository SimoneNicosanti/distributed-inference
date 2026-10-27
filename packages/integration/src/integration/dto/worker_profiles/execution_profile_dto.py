from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class BackendDto(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class LayerExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    layer_key: LayerKey
    execution_time_s: float
    memory_bytes: int | None


class BackendExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: BackendDto
    layer_profiles: dict[LayerKey, LayerExecutionProfileDto]


class ShapePointDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_shape_points: dict[str, tuple[int, ...]]


class ShapeExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape_point: ShapePointDto
    backend_profiles: tuple[BackendExecutionProfileDto, ...]


class ModelExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_id: WorkerId
    model_variant_id: ModelVariantId
    shape_execution_profiles: tuple[ShapeExecutionProfileDto, ...]
