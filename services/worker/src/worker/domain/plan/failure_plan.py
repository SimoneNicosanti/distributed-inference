from dataclasses import dataclass

from shared.flow.flow import FlowId
from shared.plan.hop_logical_id import PlannedSinkReplicaId


class NoSinkReplicaIdForFlowError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class FailurePlan:
    flow_ids: tuple[FlowId, ...]
    sink_replica_ids: tuple[PlannedSinkReplicaId, ...]

    def __post_init__(self) -> None:
        if len(self.flow_ids) != len(self.sink_replica_ids):
            raise ValueError("Each flow must have a failure sink")
        if len(self.flow_ids) != len(set(self.flow_ids)):
            raise ValueError("Failure plan flows must be unique")

    def get_sink_replica_id_for_flow(self, flow_id: FlowId) -> PlannedSinkReplicaId:
        try:
            index = self.flow_ids.index(flow_id)
            return self.sink_replica_ids[index]
        except ValueError as e:
            raise NoSinkReplicaIdForFlowError from e
