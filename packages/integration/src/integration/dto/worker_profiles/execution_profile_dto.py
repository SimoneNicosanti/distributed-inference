from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from integration.dto.model.profiled_model_variant_dto import ModelInputConcreteShapesDto
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class BackendDto(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class LayerExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_time_s: float = Field(strict=True, ge=0, allow_inf_nan=False)
    memory_bytes: Annotated[int, Field(strict=True, ge=0)] | None


class BackendExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: BackendDto
    layer_profiles: dict[LayerKey, LayerExecutionProfileDto]


class ShapeExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_shapes: ModelInputConcreteShapesDto
    backend_profiles: tuple[BackendExecutionProfileDto, ...]


class ModelExecutionProfileDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_id: WorkerId
    model_variant_id: ModelVariantId
    shape_execution_profiles: tuple[ShapeExecutionProfileDto, ...]
