from functools import total_ordering
from typing import Any

from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.identifiers import WorkerId
from distributed_inference.model_manager.domain.sub_model import (
    SubModelId,
)


@total_ordering
class InferencePlanVersion(BaseModel):
    model_config = ConfigDict(frozen=True)

    version_number: int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, InferencePlanVersion):
            return NotImplemented
        return self.version_number < other.version_number


class ResourceAllocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    use_gpu: bool


## This is the deployment of a sub-model on a worker service
## The deployment is uniquely identified by:
## - the sub-model id
## - the worker it is deployed on
## - the allocated resources
## - the replica idx (unique within the deployment)
##   - Replica idx allows to distinguish between multiple replicas of the same sub-model, same worker, same resources
##   - As such this index is scoped by the tuple (sub-model-id, worker-id, resource-allocation)
## In this way, we can also avoid rebuild of executors: if the deployment has not change we already have everything we need
## TODO: Use an hash of allocated resources to distinguish between deployments!!
class SubModelDeployment(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_model_id: SubModelId
    worker_id: WorkerId
    resource_allocation: ResourceAllocation
    replica_idx: int


class ServiceInferencePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: InferencePlanVersion
    worker_id: WorkerId

    sub_model_deployments: list[SubModelDeployment]


class WholeInferencePlan(BaseModel):
    model_config = ConfigDict(frozen=True)
