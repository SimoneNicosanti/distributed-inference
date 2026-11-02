from dataclasses import dataclass
from typing import TypeAlias

from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)


@dataclass(frozen=True, slots=True)
class ExternalHopId:
    pass


PlannedExecutableHopId: TypeAlias = PlannedPartitionReplicaId  # noqa: UP040

PlannedSourceHopId: TypeAlias = (  # noqa: UP040
    ExternalHopId | PlannedIngressReplicaId | PlannedPartitionReplicaId
)
PlannedDestinationHopId: TypeAlias = (  # noqa: UP040
    PlannedIngressReplicaId | PlannedPartitionReplicaId | PlannedSinkReplicaId
)

PlannedInvocableHopId: TypeAlias = PlannedDestinationHopId  # noqa: UP040

PlannedHopId: TypeAlias = ExternalHopId | PlannedInvocableHopId  # noqa: UP040
