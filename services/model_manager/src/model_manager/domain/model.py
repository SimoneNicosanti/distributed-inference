from dataclasses import dataclass

from shared.model.model import (
    ArchitectureInfo,
    ModelId,
    ModelTask,
    ModelType,
)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    model_task: ModelTask
    architecture_info: ArchitectureInfo

    @property
    def model_type(self) -> ModelType:
        return self.architecture_info.kind


# A model groups variants for a specific task and model type.
# It can have multiple variants, each with a specific configuration.
# For example, yolo11-cls can have yolo11-cls-n-fp32-b0 and
# yolo11-cls-x-int8-b8 variants.
@dataclass(frozen=True, slots=True)
class Model:
    model_id: ModelId
    model_info: ModelInfo

    @property
    def model_name(self) -> str:
        return self.model_id.model_name
