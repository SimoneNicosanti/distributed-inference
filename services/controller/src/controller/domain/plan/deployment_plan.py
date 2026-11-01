from dataclasses import dataclass

from controller.domain.hop.hop import ExecutableHop, InvocableHop
from shared.plan.hop_logical_id import IngressReplicaId, SinkReplicaId


@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    use_gpu: bool


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    executable_hops: tuple[ExecutableHop, ...]
    allocations: tuple[ResourceAllocation, ...]
    ingress_hops: tuple[IngressReplicaId, ...]
    sink_hops: tuple[SinkReplicaId, ...]

    def __post_init__(self) -> None:
        if len(self.executable_hops) != len(set(self.executable_hops)):
            raise ValueError("Executable hops must be unique")

        if len(self.executable_hops) != len(self.allocations):
            raise ValueError("Each executable hop must have an allocation")
        if len(self.ingress_hops) != len(set(self.ingress_hops)):
            raise ValueError("Ingress hops must be unique")
        if len(self.sink_hops) != len(set(self.sink_hops)):
            raise ValueError("Sink hops must be unique")

    @property
    def deployed_hops(self) -> tuple[InvocableHop, ...]:
        return (*self.executable_hops, *self.ingress_hops, *self.sink_hops)
