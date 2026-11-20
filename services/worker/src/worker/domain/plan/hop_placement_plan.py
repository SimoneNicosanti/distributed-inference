from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.plan.hop import PlannedInvocableHopId
from shared.service.service import WorkerId


class NoPlacementForHopError(Exception): ...


@dataclass(frozen=True, slots=True)
class HopPlacementPlan:
    mapping: Mapping[PlannedInvocableHopId, WorkerId]

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping", MappingProxyType(dict(self.mapping)))

    def get_worker_id_for_hop(self, hop_id: PlannedInvocableHopId) -> WorkerId:
        try:
            return self.mapping[hop_id]
        except KeyError as e:
            raise NoPlacementForHopError(f"No planned placement for {hop_id}") from e
