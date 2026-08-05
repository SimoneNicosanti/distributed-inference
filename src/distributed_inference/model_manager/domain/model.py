from enum import StrEnum, auto

from pydantic import BaseModel, ConfigDict, field_validator

from distributed_inference.domain.identifiers import UserId


class ModelTask(StrEnum):
    CLASSIFICATION = auto()
    DETECTION = auto()
    SEGMENTATION = auto()

    REGRESSION = auto()


class ModelType(StrEnum):
    CNN = auto()
    VIT = auto()
    BERT = auto()


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_task: ModelTask
    model_type: ModelType


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


## A model represents a group of model versions all handling a
# specific task and with a specific model type. A model is:
## - Owned by a specific user that declares visibility for it.
## - Can have multiple versions each with a specific configuration.
## For example, we can consider a yolo11-cls model owned by the system
## Possible multiple versions: yolo11-cls-n-fp32-b0 or yolo11-cls-x-int8-b8, which are:
## - nano, fp32, dynamic batch
## - xlarge, quantized int8, static batch size 8
## When asking, the user can use all the versions of the models he owns and all the public models
class Model(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: ModelId
    visibility: ModelVisibility

    model_info: ModelInfo

    @property
    def model_name(self) -> str:
        return self.model_id.model_name

    @property
    def owner_id(self) -> UserId:
        return self.model_id.owner_id
