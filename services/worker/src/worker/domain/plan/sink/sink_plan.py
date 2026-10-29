from dataclasses import dataclass

from shared.flow.flow import FlowId
from shared.model.keys import TensorKey
from shared.plan.plan_version import PlanVersion


@dataclass(frozen=True, slots=True)
class SinkPlan:
    plan_version: PlanVersion
    flow_ids: tuple[FlowId, ...]
    result_tensors_by_flow: tuple[tuple[TensorKey, ...], ...]

    def __post_init__(self) -> None:
        if not self.flow_ids:
            raise ValueError("Sink plan must not be empty")

        if len(self.flow_ids) != len(set(self.flow_ids)):
            raise ValueError("Sink plan flow ids must be unique")

        if len(self.flow_ids) != len(self.result_tensors_by_flow):
            raise ValueError("Each sink flow must define its result tensors")

        for result_tensors in self.result_tensors_by_flow:
            if not result_tensors:
                raise ValueError("Sink flow result tensors must not be empty")

            if len(result_tensors) != len(set(result_tensors)):
                raise ValueError("Sink flow result tensors must be unique")

    def get_result_tensors_by_flow_id(
        self,
        flow_id: FlowId,
    ) -> tuple[TensorKey, ...]:
        try:
            flow_index = self.flow_ids.index(flow_id)
        except ValueError:
            raise ValueError(f"Flow {flow_id} not found in sink plan") from None

        return self.result_tensors_by_flow[flow_index]
