from dataclasses import dataclass

from shared.plan.hop_logical_id import (
    IngressReplicaId,
    PartitionReplicaId,
    SinkReplicaId,
)


@dataclass(frozen=True, slots=True)
class ExternalHop:
    pass


type ExecutableHop = PartitionReplicaId

type SourceHop = ExternalHop | IngressReplicaId | PartitionReplicaId
type DestinationHop = IngressReplicaId | PartitionReplicaId | SinkReplicaId

type InvocableHop = DestinationHop

type Hop = ExternalHop | InvocableHop
