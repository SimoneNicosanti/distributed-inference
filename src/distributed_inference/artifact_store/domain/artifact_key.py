from enum import StrEnum, auto
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from distributed_inference.model_manager.domain.model_version import ModelVersionId
from distributed_inference.model_manager.domain.sub_model import SubModelId


class ArtifactKind(StrEnum):
    SUB_MODEL = auto()
    MODEL_VERSION = auto()


type ArtifactKey = Annotated[
    SubModelArtifactKey | ModelVersionArtifactKey,
    Field(discriminator="kind"),
]


class SubModelArtifactKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal[ArtifactKind.SUB_MODEL] = ArtifactKind.SUB_MODEL

    id: SubModelId


class ModelVersionArtifactKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal[ArtifactKind.MODEL_VERSION] = ArtifactKind.MODEL_VERSION

    id: ModelVersionId
