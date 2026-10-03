from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class UserId(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)


SYSTEM_USER_ID = UserId(id=UUID("8db917c1-2494-4b25-a79c-12f97cb67942"))
