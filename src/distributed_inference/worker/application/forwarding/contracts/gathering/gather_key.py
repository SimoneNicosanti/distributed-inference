from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.identifiers import SubModelDeploymentId
from distributed_inference.worker.domain.model_pass.model_pass_context import (
    ModelPassContext,
)


class GatherKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContext
    sub_model_deployment_id: SubModelDeploymentId
