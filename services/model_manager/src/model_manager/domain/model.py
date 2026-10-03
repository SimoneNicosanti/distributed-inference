from pydantic import BaseModel, ConfigDict

from shared.model.model import (
    ArchitectureInfo,
    ModelId,
    ModelTask,
    ModelType,
    ModelVisibility,
)
from shared.user.user import UserId


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_task: ModelTask
    architecture_info: ArchitectureInfo

    @property
    def model_type(self) -> ModelType:
        return self.architecture_info.kind


## A model represents a group of model variants all handling a
# specific task and with a specific model type. A model is:
## - Owned by a specific user that declares visibility for it.
## - Can have multiple variants each with a specific configuration.
## For example, we can consider a yolo11-cls model owned by the system
## Possible multiple variants: yolo11-cls-n-fp32-b0 or yolo11-cls-x-int8-b8, which are:
## - nano, fp32, dynamic batch
## - xlarge, quantized int8, static batch size 8
## When asking, the user can use all the variants of the models he owns and all the public models
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
