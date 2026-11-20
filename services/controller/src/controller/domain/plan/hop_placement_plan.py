from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from controller.domain.hop.hop import InvocableHop
from shared.service.service import WorkerId


@dataclass(frozen=True, slots=True)
class HopPlacementPlan:
    mapping: Mapping[InvocableHop, WorkerId]

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping", MappingProxyType(dict(self.mapping)))
