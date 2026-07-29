from enum import StrEnum, auto
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

type ResourceAvailability = dict[ResourceType, float]
type ResourceLock = dict[ResourceType, ResourceRequirement]


class ResourceType(StrEnum):
    COMPUTE = auto()
    MEMORY = auto()
    NETWORK = auto()


class ResourceRequirement(BaseModel):
    model_config = ConfigDict(frozen=True)
    quantity: float
    exclusive: bool

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.exclusive:
            if self.quantity != 0:
                raise ValueError(
                    "An exclusive resource requirement must have quantity equal to 0"
                )
        else:
            if self.quantity <= 0:
                raise ValueError("Quantity must be positive")

        return self


class ResourceLeaseId(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID = Field(default_factory=uuid4)


class ResourceLease(BaseModel):
    model_config = ConfigDict(frozen=True)
    resource_lease_id: ResourceLeaseId
