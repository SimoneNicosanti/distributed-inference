from dataclasses import dataclass, field
from uuid import UUID, uuid4

from front_end.domain.tensor_map import TensorMap
from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    flow_id: FlowId
    plan_version: PlanVersion
    bundle: TensorMap

    request_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    bundle: TensorMap
    request_id: UUID = field(default_factory=uuid4)
