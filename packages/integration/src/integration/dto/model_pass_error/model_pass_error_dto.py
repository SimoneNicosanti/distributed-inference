from pydantic import BaseModel, ConfigDict

from integration.dto.context.context_dto import ModelPassContextDto


class ModelPassErrorDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: ModelPassContextDto
    message: str
