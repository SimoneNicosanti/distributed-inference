from enum import StrEnum, auto
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.user.user import UserId


class ModelTask(StrEnum):
    CLASSIFICATION = auto()
    DETECTION = auto()
    SEGMENTATION = auto()

    REGRESSION = auto()


class ModelType(StrEnum):
    CNN = auto()
    VIT = auto()
    BERT = auto()


class CNNArchitectureInfo(BaseModel):
    kind: Literal[ModelType.CNN] = ModelType.CNN


class TransformerArchitectureInfo(BaseModel):
    kind: Literal[ModelType.VIT, ModelType.BERT]
    num_heads: int
    hidden_size: int


class VITArchitectureInfo(TransformerArchitectureInfo):
    kind: Literal[ModelType.VIT] = ModelType.VIT


class BERTArchitectureInfo(TransformerArchitectureInfo):
    kind: Literal[ModelType.BERT] = ModelType.BERT


type ArchitectureInfo = Annotated[
    CNNArchitectureInfo | VITArchitectureInfo | BERTArchitectureInfo,
    Field(discriminator="kind"),
]


class ModelVisibility(StrEnum):
    PUBLIC = auto()
    PRIVATE = auto()


class ModelId(BaseModel):
    model_config = ConfigDict(frozen=True)

    owner_id: UserId
    model_name: str

    @field_validator("model_name", mode="before")
    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        if model_name.find("/") != -1:
            raise ValueError("Model name cannot contain '/'")
        if model_name.find("\\") != -1:
            raise ValueError("Model name cannot contain '\\'")
        if model_name.find("..") != -1:
            raise ValueError("Model name cannot contain '..'")

        return model_name
