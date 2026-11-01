from dataclasses import dataclass

from controller.domain.hop.hop import DestinationHop, Hop, InvocableHop, SourceHop
from shared.flow.flow import FlowId
from shared.model.keys import TensorKey


@dataclass(frozen=True, slots=True)
class HopAddress[HopT: Hop]:
    flow_id: FlowId
    hop: HopT


@dataclass(frozen=True, slots=True)
class HopInputRouting:
    source: HopAddress[SourceHop]
    tensors: tuple[TensorKey, ...]

    def __post_init__(self) -> None:
        if not self.tensors or len(self.tensors) != len(set(self.tensors)):
            raise ValueError("Input routing tensors must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class HopOutputRouting:
    destination: HopAddress[DestinationHop]
    tensors: tuple[TensorKey, ...]

    def __post_init__(self) -> None:
        if not self.tensors or len(self.tensors) != len(set(self.tensors)):
            raise ValueError("Output routing tensors must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class HopRoutings:
    input_routings: tuple[HopInputRouting, ...]
    output_routings: tuple[HopOutputRouting, ...]

    def __post_init__(self) -> None:
        sources = [routing.source for routing in self.input_routings]
        destinations = [routing.destination for routing in self.output_routings]
        if len(sources) != len(set(sources)):
            raise ValueError("Input routing sources must be unique")
        if len(destinations) != len(set(destinations)):
            raise ValueError("Output routing destinations must be unique")
        tensors = [
            tensor for routing in self.input_routings for tensor in routing.tensors
        ]
        if len(tensors) != len(set(tensors)):
            raise ValueError("Input tensors must have a single source")


@dataclass(frozen=True, slots=True)
class HopRoutingPlan:
    hop_addresses: tuple[HopAddress[InvocableHop], ...]
    routings: tuple[HopRoutings, ...]

    def __post_init__(self) -> None:
        if len(self.hop_addresses) != len(set(self.hop_addresses)):
            raise ValueError("Hop addresses must be unique")
        if len(self.hop_addresses) != len(self.routings):
            raise ValueError("Each hop address must have a routing")
