from pydantic import BaseModel, ConfigDict

from shared.plan.plan import PartitionDeployment
from worker.domain.context.model_pass_context import (
    ModelPassContext,
)


class PartitionInvocationCollectionKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContext
    partition_deployment_id: PartitionDeployment
