from pydantic import BaseModel, ConfigDict

from shared.plan.partition_replica_id import PartitionReplicaId
from shared.service.service import WorkerId


class PartitionDeploymentId(BaseModel):
    model_config = ConfigDict(frozen=True)

    replica_id: PartitionReplicaId
    worker_id: WorkerId
