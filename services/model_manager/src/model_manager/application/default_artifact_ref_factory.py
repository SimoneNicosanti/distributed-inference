import json
from typing import override

from model_manager.application.abc.artifact_ref_factory import ArtifactRefFactory
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId


class DefaultArtifactRefFactory(ArtifactRefFactory):
    @override
    async def build_artifact_ref(
        self, value: ModelVariantId | ModelPartitionId
    ) -> ArtifactRef:

        if isinstance(value, ModelVariantId):
            return ArtifactRef(value=self._model_variant_artifact_seed(value))
        elif isinstance(value, ModelPartitionId):
            return ArtifactRef(value=self._model_partition_artifact_seed(value))
        else:
            raise ValueError("Invalid value type")

    @staticmethod
    def _model_variant_artifact_seed(model_variant_id: ModelVariantId) -> str:
        model_id = model_variant_id.model_id
        return json.dumps(
            [
                model_id.model_name,
                model_variant_id.variant_tag,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _model_partition_artifact_seed(partition_id: ModelPartitionId) -> str:
        return json.dumps(
            [
                DefaultArtifactRefFactory._model_variant_artifact_seed(
                    partition_id.model_variant_id
                ),
                *partition_id.layers,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
