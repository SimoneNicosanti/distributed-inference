from pydantic import BaseModel, ConfigDict

from model_manager.domain.model_variant import ModelVariant
from model_manager.domain.model_variant_profile import ModelVariantProfile
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model_variant import ModelVariantId


class ProfiledModelVariant(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_variant: ModelVariant
    profile: ModelVariantProfile
    artifact_ref: ArtifactRef

    @property
    def model_variant_id(self) -> ModelVariantId:
        return self.model_variant.model_variant_id
