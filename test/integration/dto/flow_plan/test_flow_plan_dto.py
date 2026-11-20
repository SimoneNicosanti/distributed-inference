import pytest
from pydantic import ValidationError

from integration.dto.flow_plan.flow_plan_dto import (
    FlowPlanDto,
    InputSpecDto,
    ModelSpecsDto,
)
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion
from shared.service.service import ServiceId


def _flow_plan(
    shape: tuple[int | str | None, ...], *, named_workers: bool = False
) -> FlowPlanDto:
    variant = ModelVariantId(ModelId(model_name="model"), "fp32")
    return FlowPlanDto(
        plan_version=PlanVersion(version_number=3),
        model_specs=ModelSpecsDto(
            model_variant_id=variant,
            inputs=(InputSpecDto(name="input", shape=shape, dtype="float32"),),
        ),
        ingress_replica_id=PlannedIngressReplicaId(variant, replica_idx=1),
        ingress_worker_id=ServiceId(id="ingress-worker")
        if named_workers
        else ServiceId(),
        sink_replica_id=PlannedSinkReplicaId(variant, replica_idx=2),
        sink_worker_id=ServiceId(id="sink-worker") if named_workers else ServiceId(),
    )


@pytest.mark.parametrize("shape", [(), (1, 3, 224, 224), ("batch", 3, None), (0,)])
@pytest.mark.parametrize("named_workers", [False, True])
def test_flow_plan_round_trips_json(
    shape: tuple[int | str | None, ...], named_workers: bool
) -> None:
    dto = _flow_plan(shape, named_workers=named_workers)

    restored = FlowPlanDto.model_validate_json(dto.model_dump_json())

    assert restored == dto
    assert restored.model_specs.inputs[0].shape == shape
    assert restored.model_specs.inputs[0].dtype == "float32"
    assert restored.plan_version == PlanVersion(version_number=3)
    assert restored.ingress_replica_id == dto.ingress_replica_id
    assert restored.ingress_worker_id == dto.ingress_worker_id
    assert restored.sink_replica_id == dto.sink_replica_id
    assert restored.sink_worker_id == dto.sink_worker_id


@pytest.mark.parametrize("field", ["ingress_worker_id", "sink_worker_id"])
def test_flow_plan_requires_endpoint_workers(field: str) -> None:
    payload = _flow_plan((1,)).model_dump(mode="json")
    payload.pop(field)

    with pytest.raises(ValidationError) as caught:
        FlowPlanDto.model_validate(payload)

    assert any(
        error["loc"] == (field,) and error["type"] == "missing"
        for error in caught.value.errors()
    )


def test_nested_flow_plan_dtos_are_frozen() -> None:
    dto = _flow_plan((1,))

    for model, attribute in (
        (dto, "plan_version"),
        (dto, "ingress_worker_id"),
        (dto, "sink_worker_id"),
        (dto.model_specs, "inputs"),
        (dto.model_specs.inputs[0], "dtype"),
    ):
        with pytest.raises(ValidationError, match="frozen"):
            setattr(model, attribute, getattr(model, attribute))
