from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Literal


class ModelTask(StrEnum):
    CLASSIFICATION = auto()
    DETECTION = auto()
    SEGMENTATION = auto()

    REGRESSION = auto()


class ModelType(StrEnum):
    CNN = auto()
    VIT = auto()
    BERT = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class CNNArchitectureInfo:
    kind: Literal[ModelType.CNN] = ModelType.CNN


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformerArchitectureInfo:
    kind: Literal[ModelType.VIT, ModelType.BERT]
    num_heads: int
    hidden_size: int


@dataclass(frozen=True, slots=True, kw_only=True)
class VITArchitectureInfo(TransformerArchitectureInfo):
    kind: Literal[ModelType.VIT] = ModelType.VIT


@dataclass(frozen=True, slots=True, kw_only=True)
class BERTArchitectureInfo(TransformerArchitectureInfo):
    kind: Literal[ModelType.BERT] = ModelType.BERT


type ArchitectureInfo = CNNArchitectureInfo | VITArchitectureInfo | BERTArchitectureInfo


@dataclass(frozen=True, slots=True)
class ModelId:
    model_name: str

    def __post_init__(self) -> None:
        if self.model_name.find("/") != -1:
            raise ValueError("Model name cannot contain '/'")
        if self.model_name.find("\\") != -1:
            raise ValueError("Model name cannot contain '\\'")
        if self.model_name.find("..") != -1:
            raise ValueError("Model name cannot contain '..'")
