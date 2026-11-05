from dataclasses import dataclass

from shared.artifact.artifact_ref import ArtifactRef
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from worker.domain.profiling.model_execution.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)


@dataclass(frozen=True, slots=True)
class ModelStaticProfile:
    model_variant_id: ModelVariantId
    artifact_ref: ArtifactRef
    input_shape_configurations: tuple[ModelInputConcreteShapes, ...]

    contractions: tuple[tuple[LayerKey, ...], ...]
