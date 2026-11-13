from dataclasses import dataclass

from shared.service.service import WorkerId


@dataclass(frozen=True, slots=True)
class FixedPlacement:
    worker_id: WorkerId


@dataclass(frozen=True, slots=True)
class AnyPlacement:
    pass


type Placement = FixedPlacement | AnyPlacement


@dataclass(frozen=True, slots=True)
class FlowEndpointPlacement:
    ingress_placement: Placement
    sink_placement: Placement
