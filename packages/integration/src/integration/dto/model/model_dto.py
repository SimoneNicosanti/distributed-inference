from pydantic import BaseModel

from shared.model.model import ArchitectureInfo, ModelId, ModelTask


class ModelDto(BaseModel):
    model_id: ModelId
    model_task: ModelTask
    architecture_info: ArchitectureInfo
