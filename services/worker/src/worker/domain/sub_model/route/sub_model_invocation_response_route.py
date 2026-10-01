from pydantic import BaseModel, ConfigDict

from shared.plan.plan import SubModelDeployment
from worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
)


class SubModelInvocationResponseRoute(BaseModel):
    model_config = ConfigDict(frozen=True)

    next_sub_model_deployment: SubModelDeployment

    sub_model_invocation_message: SubModelInvocationMessage
