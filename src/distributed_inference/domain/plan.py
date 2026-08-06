from functools import total_ordering
from typing import Any

from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.identifiers import ServiceId
from distributed_inference.model_manager.domain.sub_model import (
    SubModelReplicaId,
)


@total_ordering
class InferencePlanVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    version_number: int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, InferencePlanVersion):
            return NotImplemented
        return self.version_number < other.version_number


class ServiceInferencePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: InferencePlanVersion
    service_id: ServiceId

    sub_model_replicas: list[SubModelReplicaId]
    deployment_options: dict[SubModelReplicaId, DeploymentOptions]
    priorities: dict[SubModelReplicaId, int]


class WholeInferencePlan(BaseModel):
    model_config = ConfigDict(frozen=True)


class DeploymentOptions:
    use_gpu: bool
    pass
