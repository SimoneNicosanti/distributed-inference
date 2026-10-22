from uuid import UUID

from pydantic import BaseModel, ConfigDict


class WorkerIdentityDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    service_id: UUID
