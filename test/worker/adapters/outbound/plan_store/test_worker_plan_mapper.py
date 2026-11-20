from uuid import UUID

import pytest
from pydantic import ValidationError

from integration.dto.context.context_dto import (
    IngressHopDto,
    InvocableHopDto,
    PartitionReplicaHopDto,
    SinkHopDto,
)
from integration.dto.worker_plan.worker_plan_dto import (
    DeploymentPlanDto,
    FailurePlanDto,
    HopPlacementPlanDto,
    HopRoutingPlanDto,
    PartitionExecutionPlanDto,
    SchedulingPlanDto,
    WorkerPlanDto,
)
from shared.model.model import ModelId
from shared.model.model_partition import ModelPartitionId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import (
    PlannedIngressReplicaId,
    PlannedPartitionReplicaId,
    PlannedSinkReplicaId,
)
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId
from worker.adapters.outbound.plan_reader.worker_plan_mapper import worker_plan_from_dto


def _hops() -> tuple[InvocableHopDto, ...]:
    variant = ModelVariantId(ModelId(model_name="model"), "fp32")
    return (
        IngressHopDto(ingress_replica_id=PlannedIngressReplicaId(variant, 0)),
        PartitionReplicaHopDto(
            partition_replica_id=PlannedPartitionReplicaId(
                ModelPartitionId(variant, ("layer",)), 0
            )
        ),
        SinkHopDto(sink_replica_id=PlannedSinkReplicaId(variant, 0)),
    )


def _worker_plan_dto(placement: HopPlacementPlanDto) -> WorkerPlanDto:
    return WorkerPlanDto(
        plan_version=PlanVersion(version_number=3),
        deployment_plan=DeploymentPlanDto(
            executable_hops=(), allocations=(), ingress_hops=(), sink_hops=()
        ),
        scheduling_plan=SchedulingPlanDto(priority_keys=(), priorities=()),
        partition_execution_plan=PartitionExecutionPlanDto(
            partition_ids=(), schemes=()
        ),
        hop_routing_plan=HopRoutingPlanDto(hop_addresses=(), routings=()),
        failure_plan=FailurePlanDto(flow_ids=(), sink_replica_ids=()),
        hop_placement_plan=placement,
    )


@pytest.mark.parametrize("empty", [False, True])
def test_json_round_trip_preserves_hop_placement(empty: bool) -> None:
    hops = () if empty else _hops()
    workers = (
        ()
        if empty
        else (
            ServiceId(id="ingress-worker"),
            ServiceId(id=UUID("00000000-0000-0000-0000-000000000001")),
            ServiceId(id="sink-worker"),
        )
    )
    dto = _worker_plan_dto(HopPlacementPlanDto(hops=hops, worker_ids=workers))

    received = WorkerPlanDto.model_validate_json(dto.model_dump_json())
    plan = worker_plan_from_dto(received)

    assert received.model_dump(mode="json") == dto.model_dump(mode="json")
    assert plan.plan_version == dto.plan_version
    if empty:
        assert dict(plan.hop_placement_plan.mapping) == {}
    else:
        variant = ModelVariantId(ModelId(model_name="model"), "fp32")
        assert dict(plan.hop_placement_plan.mapping) == {
            PlannedIngressReplicaId(variant, 0): ServiceId(id="ingress-worker"),
            PlannedPartitionReplicaId(
                ModelPartitionId(variant, ("layer",)), 0
            ): ServiceId(id=UUID("00000000-0000-0000-0000-000000000001")),
            PlannedSinkReplicaId(variant, 0): ServiceId(id="sink-worker"),
        }


@pytest.mark.parametrize("hop_count, worker_count", [(0, 1), (1, 0)])
def test_placement_rejects_missing_or_extra_workers(
    hop_count: int, worker_count: int
) -> None:
    with pytest.raises(ValidationError, match="Each hop must have a worker"):
        HopPlacementPlanDto(
            hops=_hops()[:hop_count],
            worker_ids=(ServiceId(id="worker"),)[:worker_count],
        )


@pytest.mark.parametrize("hop", _hops())
def test_placement_rejects_duplicate_hops(hop: InvocableHopDto) -> None:
    with pytest.raises(ValidationError, match="Placement plan hops must be unique"):
        HopPlacementPlanDto(
            hops=(hop, hop),
            worker_ids=(ServiceId(id="worker-a"), ServiceId(id="worker-b")),
        )


def test_worker_plan_requires_hop_placement() -> None:
    dto = _worker_plan_dto(HopPlacementPlanDto(hops=(), worker_ids=()))
    payload = dto.model_dump(mode="json", exclude={"hop_placement_plan"})

    with pytest.raises(ValidationError, match="hop_placement_plan"):
        WorkerPlanDto.model_validate(payload)


def test_placement_rejects_external_hops() -> None:
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        HopPlacementPlanDto.model_validate(
            {
                "hops": [{"kind": "external"}],
                "worker_ids": [{"id": "worker"}],
            }
        )
