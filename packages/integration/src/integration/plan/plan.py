from functools import total_ordering
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from integration.flow.flow import FlowId
from integration.identifiers.identifiers import WorkerId
from integration.model.sub_model import (
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


class SubModelExecutionScheme(BaseModel):
    model_config = ConfigDict(frozen=True)

    inputs: list[str]
    outputs: list[str]

    @model_validator(mode="after")
    def validate_scheme(self) -> Self:
        if len(self.inputs) != len(set(self.inputs)):
            raise ValueError("Inputs must be unique")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError("Outputs must be unique")
        return self


class SubModelFollowers(BaseModel):
    model_config = ConfigDict(frozen=True)

    follower_id: SubModelDeployment
    tensors_to_pass: list[str]


class PriorityKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    flow_id: FlowId
    sub_model_deployment: SubModelDeployment


type PriorityValue = int


class ServiceInferencePlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    plan_version: InferencePlanVersion
    worker_id: WorkerId

    sub_model_deployments: list[SubModelDeployment]
    sub_model_execution_schemes: dict[SubModelId, SubModelExecutionScheme]
    sub_model_execution_topology: dict[SubModelDeployment, list[SubModelFollowers]]
    priorities: dict[PriorityKey, PriorityValue]

    @model_validator(mode="after")
    def validate_input_output_coherence(self) -> Self:

        ## TODO: Add check for coherence between sub-model execution schemes and sub-model execution topology

        return self

    def get_priority(
        self,
        flow_id: FlowId,
        sub_model_deployment: SubModelDeployment,
    ) -> PriorityValue:
        priority_key = PriorityKey(
            flow_id=flow_id,
            sub_model_deployment=sub_model_deployment,
        )
        if priority_key not in self.priorities:
            raise ValueError(f"No priority for {priority_key}")
        return self.priorities[priority_key]
