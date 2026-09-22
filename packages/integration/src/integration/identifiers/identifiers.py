from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# We can have multiple users.
# - Each user can define multiple flows.
# - Each flow can specify the model (or task type) to be executed.
# - Each model has multiple versions.
# - Each model version can be divided in multiple components after the optimization
# - Then we have the artifacts as stored in the model store.


class ServerId(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)


class ServiceId(BaseModel):
    model_config = ConfigDict(frozen=True)

    server_id: ServerId
    service_id: UUID = Field(default_factory=uuid4)


type WorkerId = ServiceId


class UserId(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)


SYSTEM_USER_ID = UserId(id=UUID("8db917c1-2494-4b25-a79c-12f97cb67942"))
