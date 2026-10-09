from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from ingress.domain.model_pass_result import ModelPassResult
from shared.context.model_invocation_context import ModelInvocationContext
from shared.context.model_pass_context import ModelPassType
from shared.plan.plan_version import PlanVersion
from shared.tensor.tensor_bundle import TensorBundle


class ModelPassRequestInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_invocation_context: ModelInvocationContext
    plan_version: PlanVersion
    model_pass_type: ModelPassType


class ModelPassCoordinator(ABC):
    @abstractmethod
    async def run_model_pass(
        self, model_pass_request_info: ModelPassRequestInfo, bundle: TensorBundle
    ) -> ModelPassResult: ...
