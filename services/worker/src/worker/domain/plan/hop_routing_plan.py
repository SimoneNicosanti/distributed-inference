from dataclasses import dataclass

from shared.flow.flow import FlowId
from shared.model.keys import TensorKey
from worker.domain.hop.hop import DestinationHop, Hop, InvocableHop, SourceHop


@dataclass(frozen=True, slots=True)
class HopAddress[HopT: Hop]:
    flow_id: FlowId
    hop: HopT


@dataclass(frozen=True, slots=True)
class HopInputRouting:
    source: HopAddress[SourceHop]
    tensors: tuple[TensorKey, ...]


@dataclass(frozen=True, slots=True)
class HopOutputRouting:
    destination: HopAddress[DestinationHop]
    tensors: tuple[TensorKey, ...]


@dataclass(frozen=True, slots=True)
class HopRoutings:
    input_routings: tuple[HopInputRouting, ...]
    output_routings: tuple[HopOutputRouting, ...]

    def check_inputs_ready(
        self,
        arrived: dict[HopAddress[SourceHop], set[TensorKey]],
    ) -> bool:
        expected = {
            input_routing.source: set(input_routing.tensors)
            for input_routing in self.input_routings
        }

        unexpected_sources = arrived.keys() - expected.keys()
        if unexpected_sources:
            raise ValueError(f"Unexpected input sources: {unexpected_sources}")

        for source, tensors in arrived.items():
            if not tensors.issubset(expected[source]):
                raise ValueError(f"Unexpected tensors from source {source}")

        return all(
            arrived.get(source, set()) == tensors
            for source, tensors in expected.items()
        )

    def route_output_tensors(
        self,
        available_tensors: set[TensorKey],
    ) -> dict[HopAddress[DestinationHop], set[TensorKey]]:
        routed = {
            route.destination: set(route.tensors) for route in self.output_routings
        }

        missing = {
            replica_id: tensors - available_tensors
            for replica_id, tensors in routed.items()
            if not tensors.issubset(available_tensors)
        }
        if missing:
            raise ValueError(f"Missing output tensors: {missing}")

        return routed


@dataclass(frozen=True, slots=True)
class HopRoutingPlan:
    hop_addresses: tuple[HopAddress[InvocableHop], ...]
    routings: tuple[HopRoutings, ...]

    def get_routings_for_hop_address(
        self, hop_address: HopAddress[InvocableHop]
    ) -> HopRoutings:
        if hop_address not in self.hop_addresses:
            raise KeyError(f"No routing for hop address {hop_address}")
        return self.routings[self.hop_addresses.index(hop_address)]
