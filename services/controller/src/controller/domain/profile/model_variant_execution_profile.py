from pydantic import BaseModel, ConfigDict, NonNegativeFloat

from controller.domain.profile.model_variant_static_profile import ShapePoint
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class LayerExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_time_s: NonNegativeFloat


class ShapeExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape_point: ShapePoint
    layer_profiles: dict[LayerKey, LayerExecutionProfile]


class ModelVariantExecutionProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape_execution_profiles: tuple[ShapeExecutionProfile, ...]


class WorkerExecutionProfiles(BaseModel):
    model_config = ConfigDict(frozen=True)

    worker_id: WorkerId
    execution_profiles: dict[ModelVariantId, ModelVariantExecutionProfile]
