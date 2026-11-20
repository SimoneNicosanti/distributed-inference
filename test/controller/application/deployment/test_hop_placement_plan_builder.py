import pytest

from controller.adapters.outbound.plan_publishing.mappers.flow_plan_mapper import (
    FlowPlanMapper,
)
from controller.adapters.outbound.plan_publishing.mappers.worker_plan_mapper import (
    WorkerPlanMapper,
)
from controller.application.deployment.builder.endpoint_replica_id_builder import (
    EndpointReplicaIdBuilder,
)
from controller.application.deployment.builder.hop_placement_plan_builder import (
    HopPlacementPlanBuilder,
)
from controller.application.deployment.builder.partition_replica_id_builder import (
    PartitionReplicaIdBuilder,
)
from controller.application.deployment.default_solution_to_plan_converter import (
    DefaultSolutionToPlanConverter,
)
from controller.domain.backend import Backend
from controller.domain.optimization.solution.enriched_optimization_solution import (
    EnrichedOptimizationSolution,
)
from controller.domain.optimization.solution.optimization_solution import (
    ComponentSolution,
    ComponentTensorContract,
    CutSolution,
    FlowOptimizationSolution,
    OptimizationSolution,
)
from controller.domain.plan.hop_placement_plan import HopPlacementPlan
from controller.domain.profile.model_variant_static_profile import TensorProperty
from front_end.adapters.outbound.mappers.flow_plan_dto_mapper import FlowPlanDtoMapper
from integration.dto.flow_plan.flow_plan_dto import FlowPlanDto
from integration.dto.worker_plan.worker_plan_dto import WorkerPlanDto
from shared.flow.flow import FlowId
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from worker.adapters.outbound.plan_reader.worker_plan_mapper import worker_plan_from_dto


def _flow(
    ingress: ServiceId, compute: ServiceId, sink: ServiceId
) -> FlowOptimizationSolution:
    return FlowOptimizationSolution(
        flow_id=FlowId(),
        ingress_worker_id=ingress,
        sink_worker_id=sink,
        model_variant_id=ModelVariantId(ModelId("model"), "fp32"),
        components=tuple(
            ComponentSolution(
                component_index=index,
                worker_id=worker,
                backend=Backend.CPU,
                layer_keys=(layer,),
                input_layer_keys=(layer,) if index == 1 else (),
                output_layer_keys=(layer,) if index == 3 else (),
                tensor_contract=ComponentTensorContract(
                    input_tensor_keys=("input",) if index < 3 else ("output",),
                    output_tensor_keys=("input",) if index == 1 else ("output",),
                ),
                priority=1,
                mean_execution_time_s=0,
                mean_execution_queue_time_s=0,
                mean_transmission_time_s=0,
                transmission_latency_s=0,
            )
            for index, worker, layer in (
                (1, ingress, "input"),
                (2, compute, "compute"),
                (3, sink, "output"),
            )
        ),
        cuts=(CutSolution(1, 2, ("input",)), CutSolution(2, 3, ("output",))),
        mean_batch_fill_time_s=0,
        mean_response_time_s=0,
    )


def _solution() -> OptimizationSolution:
    compute = ServiceId()
    isolated = ServiceId(id="isolated")
    return OptimizationSolution(
        flow_solutions=(
            _flow(ServiceId(id="first-ingress"), compute, ServiceId(id="first-sink")),
            _flow(ServiceId(id="second-ingress"), compute, ServiceId(id="second-sink")),
            _flow(isolated, isolated, isolated),
        ),
        objective_value=0,
        is_optimal=True,
    )


def test_placement_covers_remote_hops_and_sinks_only_for_participating_flows() -> None:
    solution = _solution()
    endpoints = EndpointReplicaIdBuilder.build_all(solution)
    replicas = PartitionReplicaIdBuilder.build_all(solution)
    first, second, isolated = solution.flow_solutions
    compute = first.executable_components[0].worker_id
    unused = ServiceId(id="unused")
    workers = (
        first.ingress_worker_id,
        first.sink_worker_id,
        second.ingress_worker_id,
        second.sink_worker_id,
        compute,
        isolated.ingress_worker_id,
        unused,
    )

    plans = HopPlacementPlanBuilder.build_all(workers, solution, replicas, endpoints)

    expected = {
        flow.flow_id: {
            endpoints[flow.flow_id].ingress: flow.ingress_worker_id,
            replicas[flow.flow_id][2]: flow.executable_components[0].worker_id,
            endpoints[flow.flow_id].sink: flow.sink_worker_id,
        }
        for flow in solution.flow_solutions
    }
    for flow in (first, second):
        for worker in (flow.ingress_worker_id, flow.sink_worker_id):
            assert plans[worker].mapping == expected[flow.flow_id]
    assert plans[compute].mapping == expected[first.flow_id] | expected[second.flow_id]
    assert plans[isolated.ingress_worker_id].mapping == expected[isolated.flow_id]
    assert plans[unused].mapping == {}
    assert (
        replicas[first.flow_id][2].partition_id
        == replicas[second.flow_id][2].partition_id
    )
    assert replicas[first.flow_id][2] != replicas[second.flow_id][2]


def test_placement_is_an_immutable_snapshot() -> None:
    variant = ModelVariantId(ModelId("model"), "fp32")
    hop = PlannedIngressReplicaId(variant, 0)
    worker = ServiceId()
    mapping = {hop: worker}
    plan = HopPlacementPlan(mapping=mapping)

    mapping.clear()

    assert plan.mapping == {hop: worker}
    with pytest.raises(TypeError):
        plan.mapping[hop] = ServiceId()


def test_empty_solution_has_no_placements() -> None:
    assert (
        HopPlacementPlanBuilder.build_all((), OptimizationSolution((), 0, True), {}, {})
        == {}
    )


@pytest.mark.asyncio
async def test_built_plans_round_trip_to_worker_and_front_end_domains() -> None:
    solution = _solution()
    variant = solution.flow_solutions[0].model_variant_id
    enriched = EnrichedOptimizationSolution(
        solution, {variant: {"input": TensorProperty("float32", 4, ("batch", 3))}}
    )
    version = PlanVersion(7)

    workers, flows = await DefaultSolutionToPlanConverter().convert_solution_to_plans(
        version, enriched
    )

    owners = {
        hop: worker
        for worker, plan in workers.items()
        for hop in plan.deployment_plan.deployed_hops
    }
    for worker, plan in workers.items():
        assert all(
            owner == owners[hop]
            for hop, owner in plan.hop_placement_plan.mapping.items()
        )
        assert all(
            plan.hop_placement_plan.mapping[hop] == worker
            for hop in plan.deployment_plan.deployed_hops
        )
        assert all(
            sink in plan.hop_placement_plan.mapping
            for sink in plan.failure_plan.sink_replica_ids
        )
        dto = WorkerPlanMapper.to_worker_plan_dto(plan)
        received = WorkerPlanDto.model_validate_json(dto.model_dump_json())
        restored = worker_plan_from_dto(received)
        assert restored.plan_version == version
        assert restored.hop_placement_plan.mapping == plan.hop_placement_plan.mapping

    for flow in solution.flow_solutions:
        plan = flows[flow.flow_id]
        assert plan.ingress_worker_id == flow.ingress_worker_id
        assert plan.sink_worker_id == flow.sink_worker_id
        assert owners[plan.ingress_replica_id] == plan.ingress_worker_id
        assert owners[plan.sink_replica_id] == plan.sink_worker_id
        dto = FlowPlanMapper.to_flow_plan_dto(plan)
        received = FlowPlanDto.model_validate_json(dto.model_dump_json())
        restored = FlowPlanDtoMapper.to_flow_plan(received)
        assert restored.plan_version == version
        assert restored.ingress_replica_id == plan.ingress_replica_id
        assert restored.ingress_worker_id == plan.ingress_worker_id
        assert restored.sink_replica_id == plan.sink_replica_id
        assert restored.sink_worker_id == plan.sink_worker_id
