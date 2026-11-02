from dataclasses import dataclass

from shared.plan.hop import PlannedExecutableHopId, PlannedInvocableHopId
from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedSinkReplicaId,
)


@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    use_gpu: bool


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    executable_hops: tuple[PlannedExecutableHopId, ...]
    allocations: tuple[ResourceAllocation, ...]

    ingress_hops: tuple[PlannedIngressReplicaId, ...]
    sink_hops: tuple[PlannedSinkReplicaId, ...]

    def __post_init__(self) -> None:
        if len(self.executable_hops) != len(set(self.executable_hops)):
            raise ValueError("Executable hops must be unique")

        if len(self.executable_hops) != len(self.allocations):
            raise ValueError("Each executable hop must have an allocation")

        if len(self.ingress_hops) != len(set(self.ingress_hops)):
            raise ValueError("Ingress hops must be unique")

        if len(self.sink_hops) != len(set(self.sink_hops)):
            raise ValueError("Sink hops must be unique")

    def get_allocation_by_executable_hop(
        self, executable_hop: PlannedExecutableHopId
    ) -> ResourceAllocation:
        return self.allocations[self.executable_hops.index(executable_hop)]

    @property
    def planned_hops(self) -> tuple[PlannedInvocableHopId, ...]:
        return (*self.executable_hops, *self.ingress_hops, *self.sink_hops)
