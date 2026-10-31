from dataclasses import dataclass

from shared.plan.hop_logical_id import (
    IngressReplicaId,
    PartitionReplicaId,
    SinkReplicaId,
)
from shared.service.service import WorkerId

type HopReplicaId = PartitionReplicaId | IngressReplicaId | SinkReplicaId


@dataclass(frozen=True, slots=True)
class HopDeploymentId[ReplicaIdT: HopReplicaId]:
    replica_id: ReplicaIdT
    worker_id: WorkerId
