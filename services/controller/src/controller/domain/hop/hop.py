from dataclasses import dataclass

from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)


@dataclass(frozen=True, slots=True)
class ExternalHop:
    pass


type ExecutableHop = PlannedPartitionReplicaId
type SourceHop = ExternalHop | PlannedIngressReplicaId | PlannedPartitionReplicaId
type DestinationHop = (
    PlannedIngressReplicaId | PlannedPartitionReplicaId | PlannedSinkReplicaId
)
type InvocableHop = DestinationHop
type Hop = ExternalHop | InvocableHop
