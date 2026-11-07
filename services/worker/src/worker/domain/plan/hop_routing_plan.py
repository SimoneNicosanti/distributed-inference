from dataclasses import dataclass

from shared.flow.flow import FlowId
from shared.model.keys import TensorKey
from shared.plan.hop import (
    PlannedDestinationHopId,
    PlannedHopId,
    PlannedInvocableHopId,
    PlannedSourceHopId,
)


class NotExistentHopRoutingError(Exception):
    pass


class UnexpectedSourcesError(Exception):
    pass


class UnexpectedTensorsFromSourceError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class HopAddress[HopT: PlannedHopId]:
    flow_id: FlowId
    hop: HopT


@dataclass(frozen=True, slots=True)
class HopInputRouting:
    source: HopAddress[PlannedSourceHopId]
    tensors: tuple[TensorKey, ...]


@dataclass(frozen=True, slots=True)
class HopOutputRouting:
    destination: HopAddress[PlannedDestinationHopId]
    tensors: tuple[TensorKey, ...]


@dataclass(frozen=True, slots=True)
class HopRoutings:
    input_routings: tuple[HopInputRouting, ...]
    output_routings: tuple[HopOutputRouting, ...]

    def check_inputs_ready(
        self,
        arrived: dict[HopAddress[PlannedSourceHopId], set[TensorKey]],
    ) -> bool:
        expected = {
            input_routing.source: set(input_routing.tensors)
            for input_routing in self.input_routings
        }

        return all(
            arrived.get(source, set()) == tensors
            for source, tensors in expected.items()
        )

    def check_valid_sources(
        self, arrived: dict[HopAddress[PlannedSourceHopId], set[TensorKey]]
    ) -> None:
        expected = {
            input_routing.source: set(input_routing.tensors)
            for input_routing in self.input_routings
        }

        unexpected_sources = arrived.keys() - expected.keys()
        if unexpected_sources:
            raise UnexpectedSourcesError(
                f"Unexpected input sources: {unexpected_sources}"
            )

        for source, tensors in arrived.items():
            if not tensors.issubset(expected[source]):
                raise UnexpectedTensorsFromSourceError(
                    f"Unexpected tensors from source {source}"
                )

    def route_output_tensors(
        self,
        available_tensors: set[TensorKey],
    ) -> dict[HopAddress[PlannedDestinationHopId], set[TensorKey]]:
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
    hop_addresses: tuple[HopAddress[PlannedInvocableHopId], ...]
    routings: tuple[HopRoutings, ...]

    def get_routings_for_hop_address(
        self, hop_address: HopAddress[PlannedInvocableHopId]
    ) -> HopRoutings:
        if hop_address not in self.hop_addresses:
            raise NotExistentHopRoutingError(
                f"No routing for hop address {hop_address}"
            )
        return self.routings[self.hop_addresses.index(hop_address)]
