from abc import ABC, abstractmethod
from dataclasses import dataclass

from worker.domain.context.model_invocation_context import ModelInvocationContext
from worker.domain.context.model_pass_context import ModelPassType
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion
from worker.domain.model_pass.model_pass_result import ModelPassResult
from worker.domain.tensor.tensor_bundle import TensorBundle


@dataclass(frozen=True, slots=True)
class ModelPassRequestInfo:
    model_invocation_context: ModelInvocationContext
    plan_version: PlanVersion
    model_pass_type: ModelPassType

    @property
    def flow_id(self) -> FlowId:
        return self.model_invocation_context.flow_id


class ModelPassCoordinator(ABC):
    @abstractmethod
    async def run_model_pass(
        self, model_pass_request_info: ModelPassRequestInfo, bundle: TensorBundle
    ) -> ModelPassResult: ...
