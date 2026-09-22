from pydantic import BaseModel, ConfigDict
from integration.plan.plan import SubModelDeployment

from worker.domain.model_pass.model_pass_context import (
    ModelPassContext,
)


class GatherKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContext
    sub_model_deployment_id: SubModelDeployment
