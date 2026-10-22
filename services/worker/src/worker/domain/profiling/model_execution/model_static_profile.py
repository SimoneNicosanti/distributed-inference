from dataclasses import dataclass

from shared.artifact.artifact_ref import ArtifactRef
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from worker.domain.profiling.model_execution.shape_point import ShapePoint


@dataclass(frozen=True, slots=True)
class ModelStaticProfile:
    model_variant_id: ModelVariantId
    artifact_ref: ArtifactRef
    shape_points: tuple[ShapePoint, ...]

    contractions: tuple[tuple[LayerKey, ...], ...]
