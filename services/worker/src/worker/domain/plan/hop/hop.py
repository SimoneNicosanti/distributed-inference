from dataclasses import dataclass

from shared.plan.partition_replica_id import PartitionReplicaId
from shared.service.service import IngressId, SinkId


@dataclass(frozen=True, slots=True)
class IngressHop:
    ingress_id: IngressId


@dataclass(frozen=True, slots=True)
class ReplicaHop:
    replica_id: PartitionReplicaId


@dataclass(frozen=True, slots=True)
class SinkHop:
    sink_id: SinkId


type SourceHop = IngressHop | ReplicaHop
type DestinationHop = ReplicaHop | SinkHop

type Hop = IngressHop | ReplicaHop | SinkHop
