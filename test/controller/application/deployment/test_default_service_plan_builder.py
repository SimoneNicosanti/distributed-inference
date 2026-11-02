from collections.abc import Mapping
from itertools import pairwise

import pytest

from controller.adapters.outbound.plan_share.mappers.worker_plan_mapper import (
    WorkerPlanMapper,
)
from controller.application.deployment.default_all_worker_plan_builder import (
    DefaultSolutionToPlanConverter,
)
from controller.domain.hop.hop import ExternalHop
from controller.domain.optimization.solving.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.optimization.solving.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    CutSolution,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from controller.domain.plan.front_end_plan import FlowPlan
from controller.domain.plan.hop_routing_plan import HopAddress
from controller.domain.plan.worker_plan import WorkerPlan
from controller.domain.profile.model_variant_static_profile import TensorProperty
from integration.worker.deployment.http.service_plan_http_schema import (
    PrepareWorkerPlanRequest,
)
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop import ExternalHopId as WorkerExternalHop
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from shared.user.user import UserId
from worker.adapters.inbound.deployment.http.service_plan_mapper import (
    worker_plan_from_dto,
)


def _component(
    index: int,
    worker_id: ServiceId,
    layers: tuple[str, ...],
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    *,
    input_layers: tuple[str, ...] = (),
    output_layers: tuple[str, ...] = (),
    priority: int = 1,
) -> ComponentSolution:
    return ComponentSolution(
        component_index=index,
        worker_id=worker_id,
        layer_keys=layers,
        input_layer_keys=input_layers,
        output_layer_keys=output_layers,
        tensor_contract=ComponentTensorContract(
            input_tensor_keys=inputs,
            output_tensor_keys=outputs,
        ),
        priority=priority,
        mean_execution_time_s=0,
        mean_execution_queue_time_s=0,
        mean_transmission_time_s=0,
        transmission_latency_s=0,
    )


@pytest.mark.asyncio
async def test_maps_solution_to_all_involved_workers() -> None:
    user_id = UserId()
    model_variant_id = ModelVariantId(
        model_id=ModelId(model_name="model"),
        variant_tag="variant",
    )
    ingress_one = ServiceId()
    ingress_two = ServiceId()
    sink_one = ServiceId()
    compute_one = ServiceId()
    compute_two = ServiceId()
    first_flow_id = FlowId(user_id=user_id)
    second_flow_id = FlowId(user_id=user_id)

    first_flow = FlowOptimizationSolution(
        flow_id=first_flow_id,
        ingress_worker_id=ingress_one,
        sink_worker_id=sink_one,
        model_variant_id=model_variant_id,
        components=(
            _component(
                1,
                ingress_one,
                ("input",),
                ("x",),
                ("x",),
                input_layers=("input",),
            ),
            _component(2, compute_one, ("encoder",), ("x",), ("hidden",)),
            _component(
                3,
                compute_two,
                ("head",),
                ("hidden", "x"),
                ("result",),
                priority=2,
            ),
            _component(
                4,
                sink_one,
                ("output",),
                ("result",),
                ("result",),
                output_layers=("output",),
            ),
        ),
        cuts=(
            CutSolution(
                source_component_index=1,
                destination_component_index=2,
                tensor_keys=("x",),
            ),
            CutSolution(
                source_component_index=2,
                destination_component_index=3,
                tensor_keys=("hidden", "x"),
            ),
            CutSolution(
                source_component_index=3,
                destination_component_index=4,
                tensor_keys=("result",),
            ),
        ),
        mean_batch_fill_time_s=0,
        mean_response_time_s=0,
    )
    second_flow = FlowOptimizationSolution(
        flow_id=second_flow_id,
        ingress_worker_id=ingress_two,
        sink_worker_id=ingress_two,
        model_variant_id=model_variant_id,
        components=(
            _component(
                1,
                ingress_two,
                ("input", "encoder", "head", "output"),
                ("x",),
                ("result",),
                input_layers=("input",),
                output_layers=("output",),
                priority=3,
            ),
        ),
        cuts=(),
        mean_batch_fill_time_s=0,
        mean_response_time_s=0,
    )
    solution = OptimizationSolution(
        flow_solutions=(first_flow, second_flow),
        objective_value=0,
        is_optimal=True,
    )

    (
        plans,
        flow_plans,
    ) = await DefaultSolutionToPlanConverter().convert_solution_to_plans(
        PlanVersion(version_number=1), _enriched(solution)
    )

    assert set(plans) == {ingress_one, ingress_two, sink_one, compute_one, compute_two}
    for endpoint_worker in (ingress_one, sink_one):
        assert plans[endpoint_worker].deployment_plan.executable_hops == ()
        assert plans[endpoint_worker].deployment_plan.allocations == ()
        assert plans[endpoint_worker].partition_execution_plan.partition_ids == ()
        assert plans[endpoint_worker].scheduling_plan.priority_keys == ()
    assert len(plans[ingress_two].deployment_plan.executable_hops) == 1

    compute_one_plan = plans[compute_one]
    encoder_replicas = compute_one_plan.deployment_plan.executable_hops
    assert tuple(replica.replica_idx for replica in encoder_replicas) == (0,)
    assert compute_one_plan.partition_execution_plan.partition_ids == (
        encoder_replicas[0].partition_id,
    )
    assert compute_one_plan.scheduling_plan.priorities == (1,)
    assert compute_one_plan.partition_execution_plan.schemes[0].outputs == ("hidden",)
    assert compute_one_plan.hop_routing_plan.routings[0].output_routings[0].tensors == (
        "hidden",
        "x",
    )
    assert plans[ingress_two].scheduling_plan.priorities == (3,)
    assert plans[compute_one].deployment_plan.ingress_hops == ()
    assert plans[compute_one].deployment_plan.sink_hops == ()

    compute_two_plan = plans[compute_two]
    head_replica = compute_two_plan.deployment_plan.executable_hops[0]
    first_encoder_replica = encoder_replicas[0]
    assert (
        compute_one_plan.hop_routing_plan.routings[0].output_routings[0].destination.hop
        == head_replica
    )
    assert (
        compute_two_plan.hop_routing_plan.routings[0].input_routings[0].source.hop
        == first_encoder_replica
    )
    assert compute_two_plan.hop_routing_plan.routings[0].output_routings[0].tensors == (
        "result",
    )

    ingress_plan = plans[ingress_one]
    sink_plan = plans[sink_one]
    ingress_route = ingress_plan.hop_routing_plan.routings[0]
    sink_route = sink_plan.hop_routing_plan.routings[0]
    assert isinstance(ingress_route.input_routings[0].source.hop, ExternalHop)
    assert ingress_route.input_routings[0].source.flow_id == first_flow_id
    assert ingress_route.output_routings[0].destination.hop == first_encoder_replica
    assert ingress_route.output_routings[0].tensors == ("x",)
    assert sink_route.input_routings[0].source.hop == head_replica
    assert sink_route.input_routings[0].tensors == ("result",)
    assert sink_route.output_routings == ()
    assert (
        compute_two_plan.hop_routing_plan.routings[0].output_routings[0].destination.hop
        == sink_plan.deployment_plan.sink_hops[0]
    )
    assert ingress_plan.deployment_plan.ingress_hops[0].replica_idx == 0
    assert plans[ingress_two].deployment_plan.ingress_hops[0].replica_idx == 1
    assert sink_plan.deployment_plan.sink_hops[0].replica_idx == 0
    assert plans[ingress_two].deployment_plan.sink_hops[0].replica_idx == 1
    assert tuple(
        replica.replica_idx
        for replica in plans[ingress_two].deployment_plan.executable_hops
    ) == (0,)
    assert len(plans[ingress_two].hop_routing_plan.hop_addresses) == 3
    _assert_plan_graph(plans)
    _assert_flow_plans(plans, flow_plans)
    for plan in plans.values():
        _assert_worker_round_trip(plan)


@pytest.mark.asyncio
async def test_rejects_flow_without_executable_components() -> None:
    user_id = UserId()
    ingress_id = ServiceId()
    model_variant_id = ModelVariantId(
        model_id=ModelId(model_name="model"),
        variant_tag="variant",
    )
    solution = OptimizationSolution(
        flow_solutions=(
            FlowOptimizationSolution(
                flow_id=FlowId(user_id=user_id),
                ingress_worker_id=ingress_id,
                sink_worker_id=ingress_id,
                model_variant_id=model_variant_id,
                components=(
                    _component(
                        1,
                        ingress_id,
                        ("input",),
                        ("x",),
                        ("x",),
                        input_layers=("input",),
                    ),
                ),
                cuts=(),
                mean_batch_fill_time_s=0,
                mean_response_time_s=0,
            ),
        ),
        objective_value=0,
        is_optimal=True,
    )

    with pytest.raises(ValueError, match="has no executable components"):
        await DefaultSolutionToPlanConverter().convert_solution_to_plans(
            PlanVersion(version_number=1), _enriched(solution)
        )


def _placed_flow(
    model_variant_id: ModelVariantId,
    placements: tuple[ServiceId, ServiceId, ServiceId],
) -> FlowOptimizationSolution:
    groups: list[tuple[ServiceId, list[str]]] = []
    for worker, layer in zip(placements, ("input", "compute", "output"), strict=True):
        if groups and groups[-1][0] == worker:
            groups[-1][1].append(layer)
        else:
            groups.append((worker, [layer]))
    components = tuple(
        _component(
            index,
            worker,
            tuple(layers),
            ("y",) if layers == ["output"] else ("x",),
            ("x",) if layers == ["input"] else ("y",),
            input_layers=("input",) if "input" in layers else (),
            output_layers=("output",) if "output" in layers else (),
        )
        for index, (worker, layers) in enumerate(groups, start=1)
    )
    return FlowOptimizationSolution(
        flow_id=FlowId(user_id=UserId()),
        ingress_worker_id=placements[0],
        sink_worker_id=placements[-1],
        model_variant_id=model_variant_id,
        components=components,
        cuts=tuple(
            CutSolution(
                source.component_index,
                destination.component_index,
                source.tensor_contract.output_tensor_keys,
            )
            for source, destination in pairwise(components)
        ),
        mean_batch_fill_time_s=0,
        mean_response_time_s=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worker_indexes", [(0, 1, 2), (0, 1, 0), (0, 0, 1), (0, 1, 1), (0, 0, 0)]
)
async def test_respects_resolved_endpoint_and_compute_placements(
    worker_indexes: tuple[int, int, int],
) -> None:
    workers = (ServiceId(), ServiceId(), ServiceId())
    placements = (
        workers[worker_indexes[0]],
        workers[worker_indexes[1]],
        workers[worker_indexes[2]],
    )
    variant = ModelVariantId(ModelId(model_name="model"), "variant")
    flow = _placed_flow(variant, placements)
    (
        plans,
        flow_plans,
    ) = await DefaultSolutionToPlanConverter().convert_solution_to_plans(
        PlanVersion(version_number=1), _enriched(OptimizationSolution((flow,), 0, True))
    )

    assert set(plans) == set(placements)
    for worker_id, plan in plans.items():
        assert bool(plan.deployment_plan.ingress_hops) == (worker_id == placements[0])
        assert bool(plan.deployment_plan.sink_hops) == (worker_id == placements[2])
        assert bool(plan.deployment_plan.executable_hops) == (
            worker_id == placements[1]
        )
        assert bool(plan.partition_execution_plan.partition_ids) == (
            worker_id == placements[1]
        )
        assert bool(plan.scheduling_plan.priority_keys) == (worker_id == placements[1])
        _assert_worker_round_trip(plan)
    _assert_plan_graph(plans)
    _assert_flow_plans(plans, flow_plans)


@pytest.mark.asyncio
async def test_relay_worker_in_one_flow_can_compute_for_another_flow() -> None:
    local_worker, remote_worker = ServiceId(), ServiceId()
    variant = ModelVariantId(ModelId(model_name="model"), "variant")
    relay_flow = _placed_flow(variant, (local_worker, remote_worker, local_worker))
    inference_flow = _placed_flow(variant, (remote_worker, local_worker, remote_worker))
    (
        plans,
        flow_plans,
    ) = await DefaultSolutionToPlanConverter().convert_solution_to_plans(
        PlanVersion(version_number=1),
        _enriched(OptimizationSolution((relay_flow, inference_flow), 0, True)),
    )

    assert set(plans) == {local_worker, remote_worker}
    for worker_id, endpoint_flow, compute_flow, endpoint_index, compute_index in (
        (local_worker, relay_flow, inference_flow, 0, 1),
        (remote_worker, inference_flow, relay_flow, 1, 0),
    ):
        plan = plans[worker_id]
        assert plan.deployment_plan.ingress_hops[0].replica_idx == endpoint_index
        assert plan.deployment_plan.sink_hops[0].replica_idx == endpoint_index
        assert plan.deployment_plan.executable_hops[0].replica_idx == compute_index
        assert tuple(key.flow_id for key in plan.scheduling_plan.priority_keys) == (
            compute_flow.flow_id,
        )
        assert {
            address.flow_id
            for address in plan.hop_routing_plan.hop_addresses
            if address.hop
            in (*plan.deployment_plan.ingress_hops, *plan.deployment_plan.sink_hops)
        } == {endpoint_flow.flow_id}
        _assert_worker_round_trip(plan)
    _assert_plan_graph(plans)
    _assert_flow_plans(plans, flow_plans)


@pytest.mark.asyncio
async def test_shared_compute_worker_keeps_flow_replicas_and_deduplicates_schemes() -> (
    None
):
    first_endpoint, second_endpoint, compute = ServiceId(), ServiceId(), ServiceId()
    variant = ModelVariantId(ModelId(model_name="model"), "variant")
    first_flow = _placed_flow(variant, (first_endpoint, compute, first_endpoint))
    second_flow = _placed_flow(variant, (second_endpoint, compute, second_endpoint))
    (
        plans,
        flow_plans,
    ) = await DefaultSolutionToPlanConverter().convert_solution_to_plans(
        PlanVersion(version_number=1),
        _enriched(OptimizationSolution((first_flow, second_flow), 0, True)),
    )

    compute_plan = plans[compute]
    replicas = compute_plan.deployment_plan.executable_hops
    assert len(replicas) == 2
    assert tuple(replica.replica_idx for replica in replicas) == (0, 1)
    assert replicas[0].partition_id == replicas[1].partition_id
    assert compute_plan.partition_execution_plan.partition_ids == (
        replicas[0].partition_id,
    )
    assert len(compute_plan.partition_execution_plan.schemes) == 1
    assert tuple(key.flow_id for key in compute_plan.scheduling_plan.priority_keys) == (
        first_flow.flow_id,
        second_flow.flow_id,
    )
    assert (
        plans[first_endpoint].deployment_plan.ingress_hops
        != plans[second_endpoint].deployment_plan.ingress_hops
    )
    assert (
        plans[first_endpoint].deployment_plan.sink_hops
        != plans[second_endpoint].deployment_plan.sink_hops
    )
    _assert_plan_graph(plans)
    _assert_flow_plans(plans, flow_plans)
    for plan in plans.values():
        _assert_worker_round_trip(plan)


@pytest.mark.asyncio
async def test_empty_solution_produces_no_worker_plans() -> None:
    (
        plans,
        flow_plans,
    ) = await DefaultSolutionToPlanConverter().convert_solution_to_plans(
        PlanVersion(version_number=1), _enriched(OptimizationSolution((), 0, True))
    )
    assert plans == {}
    assert flow_plans == {}


def _enriched(solution: OptimizationSolution) -> EnrichedOptimizationSolution:
    return EnrichedOptimizationSolution(
        solution,
        {
            flow.model_variant_id: {"x": TensorProperty("float32", 4, ("batch", 3))}
            for flow in solution.flow_solutions
        },
    )


def _assert_flow_plans(
    worker_plans: Mapping[ServiceId, WorkerPlan],
    flow_plans: Mapping[FlowId, FlowPlan],
) -> None:
    addresses = {
        address
        for plan in worker_plans.values()
        for address in plan.hop_routing_plan.hop_addresses
    }
    assert set(flow_plans) == {address.flow_id for address in addresses}
    for flow_id, plan in flow_plans.items():
        assert plan.plan_version == PlanVersion(version_number=1)
        assert plan.input_tensor_properties == {
            "x": TensorProperty("float32", 4, ("batch", 3))
        }
        assert HopAddress(flow_id, plan.ingress_replica_id) in addresses
        assert HopAddress(flow_id, plan.sink_replica_id) in addresses


def _assert_plan_graph(plans: dict[ServiceId, WorkerPlan]) -> None:
    routings = {
        address: routing
        for plan in plans.values()
        for address, routing in zip(
            plan.hop_routing_plan.hop_addresses,
            plan.hop_routing_plan.routings,
            strict=True,
        )
    }
    for source, routing in routings.items():
        for output in routing.output_routings:
            destination = routings[output.destination]
            inputs = [
                binding
                for binding in destination.input_routings
                if binding.source == source
            ]
            assert len(inputs) == 1
            assert inputs[0].tensors == output.tensors
        for binding in routing.input_routings:
            if isinstance(binding.source.hop, ExternalHop):
                continue
            previous = routings[HopAddress(binding.source.flow_id, binding.source.hop)]
            outputs = [
                route
                for route in previous.output_routings
                if route.destination == source
            ]
            assert len(outputs) == 1
            assert outputs[0].tensors == binding.tensors


def _assert_worker_round_trip(plan: WorkerPlan) -> None:
    request = PrepareWorkerPlanRequest(
        worker_plan=WorkerPlanMapper.to_worker_plan_dto(plan)
    )
    received = PrepareWorkerPlanRequest.model_validate_json(request.model_dump_json())
    worker_plan = worker_plan_from_dto(received.worker_plan)
    assert worker_plan.plan_version == plan.plan_version
    assert (
        worker_plan.deployment_plan.planned_hops == plan.deployment_plan.deployed_hops
    )
    assert tuple(
        allocation.use_gpu for allocation in worker_plan.deployment_plan.allocations
    ) == tuple(allocation.use_gpu for allocation in plan.deployment_plan.allocations)
    assert worker_plan.scheduling_plan.priorities == plan.scheduling_plan.priorities
    assert tuple(
        (key.flow_id, key.executable_hop)
        for key in worker_plan.scheduling_plan.priority_keys
    ) == tuple(
        (key.flow_id, key.executable_hop) for key in plan.scheduling_plan.priority_keys
    )
    assert (
        worker_plan.partition_execution_plan.partition_ids
        == plan.partition_execution_plan.partition_ids
    )
    assert tuple(
        (scheme.inputs, scheme.outputs)
        for scheme in worker_plan.partition_execution_plan.schemes
    ) == tuple(
        (scheme.inputs, scheme.outputs)
        for scheme in plan.partition_execution_plan.schemes
    )
    assert tuple(
        (address.flow_id, address.hop)
        for address in worker_plan.hop_routing_plan.hop_addresses
    ) == tuple(
        (address.flow_id, address.hop)
        for address in plan.hop_routing_plan.hop_addresses
    )
    for actual, expected in zip(
        worker_plan.hop_routing_plan.routings,
        plan.hop_routing_plan.routings,
        strict=True,
    ):
        assert len(actual.input_routings) == len(expected.input_routings)
        for actual_input, expected_input in zip(
            actual.input_routings, expected.input_routings, strict=True
        ):
            assert actual_input.source.flow_id == expected_input.source.flow_id
            if isinstance(expected_input.source.hop, ExternalHop):
                assert isinstance(actual_input.source.hop, WorkerExternalHop)
            else:
                assert actual_input.source.hop == expected_input.source.hop
            assert actual_input.tensors == expected_input.tensors
        assert tuple(
            (route.destination.flow_id, route.destination.hop, route.tensors)
            for route in actual.output_routings
        ) == tuple(
            (route.destination.flow_id, route.destination.hop, route.tensors)
            for route in expected.output_routings
        )
