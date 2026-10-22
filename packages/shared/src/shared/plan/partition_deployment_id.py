from dataclasses import dataclass

from shared.plan.partition_replica_id import PartitionReplicaId
from shared.service.service import WorkerId


@dataclass(frozen=True, slots=True)
class PartitionDeploymentId:
    replica_id: PartitionReplicaId
    worker_id: WorkerId
