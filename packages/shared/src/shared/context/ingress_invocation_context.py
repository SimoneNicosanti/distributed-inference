from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from shared.context.model_pass_context import ModelPassContext

type IngressInvocationId = UUID


class IngressInvocationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_pass_context: ModelPassContext
    ingress_invocation_id: IngressInvocationId = Field(default_factory=uuid4)
