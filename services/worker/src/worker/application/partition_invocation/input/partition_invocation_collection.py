from dataclasses import dataclass

from shared.plan.partition_replica_id import PartitionReplicaId
from worker.domain.context.model_pass_context import ModelPassContext
from worker.domain.partition.partition_invocation_contribution import (
    PartitionInvocationContribution,
)


@dataclass(frozen=True, slots=True)
class PartitionInvocationCollectionKey:
    model_pass_context: ModelPassContext
    partition_replica_id: PartitionReplicaId


@dataclass(frozen=True, slots=True)
class PartitionInvocationCollection:
    key: PartitionInvocationCollectionKey
    contributions: tuple[PartitionInvocationContribution, ...]
