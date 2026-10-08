from pydantic import BaseModel, ConfigDict

from shared.artifact.artifact_ref import ArtifactRef
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from worker.domain.profiling.model_execution.shape_point import ShapePoint


class ModelStaticProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_variant_id: ModelVariantId
    artifact_ref: ArtifactRef
    shape_points: list[ShapePoint]

    contractions: list[tuple[LayerKey, ...]]
