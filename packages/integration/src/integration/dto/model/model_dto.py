from pydantic import BaseModel

from shared.model.model import ArchitectureInfo, ModelId, ModelTask, ModelVisibility


class ModelDto(BaseModel):
    model_id: ModelId
    visibility: ModelVisibility
    model_task: ModelTask
    architecture_info: ArchitectureInfo
