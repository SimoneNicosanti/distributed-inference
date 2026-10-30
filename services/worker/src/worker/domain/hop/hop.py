from dataclasses import dataclass

from shared.plan.partition_replica_id import PartitionReplicaId
from shared.service.service import IngressId, SinkId


@dataclass(frozen=True, slots=True)
class ExternalHop:
    pass


@dataclass(frozen=True, slots=True)
class IngressHop:
    ingress_id: IngressId


@dataclass(frozen=True, slots=True)
class PartitionReplicaHop:
    partition_replica_id: PartitionReplicaId


@dataclass(frozen=True, slots=True)
class SinkHop:
    sink_id: SinkId


type ExecutableHop = PartitionReplicaHop

type SourceHop = ExternalHop | IngressHop | ExecutableHop
type DestinationHop = IngressHop | ExecutableHop | SinkHop

type InvocableHop = DestinationHop

type Hop = ExternalHop | InvocableHop
