import pytest

from front_end.adapters.outbound.mappers.flow_plan_dto_mapper import FlowPlanDtoMapper
from front_end.domain.flow_plan import FlowPlan
from front_end.domain.model_specs import InputSpec, ModelSpecs
from integration.dto.flow_plan.flow_plan_dto import (
    FlowPlanDto,
    InputSpecDto,
    ModelSpecsDto,
)
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.plan.hop_logical_id import PlannedIngressReplicaId, PlannedSinkReplicaId
from shared.plan.plan_version import PlanVersion


def _flow_plan_dto(inputs: tuple[InputSpecDto, ...]) -> FlowPlanDto:
    variant = ModelVariantId(ModelId(model_name="model"), "fp32")
    return FlowPlanDto(
        plan_version=PlanVersion(version_number=3),
        model_specs=ModelSpecsDto(model_variant_id=variant, inputs=inputs),
        ingress_replica_id=PlannedIngressReplicaId(variant, replica_idx=1),
        sink_replica_id=PlannedSinkReplicaId(variant, replica_idx=2),
    )


@pytest.mark.parametrize("shape", [(), (1, 3, 224, 224), ("batch", 3, None), (0,)])
def test_maps_all_fields_and_preserves_input_order(
    shape: tuple[int | str | None, ...],
) -> None:
    dto = _flow_plan_dto(
        (
            InputSpecDto(name="input", shape=shape, dtype="float32"),
            InputSpecDto(name="mask", shape=(1,), dtype="bool"),
        )
    )
    received = FlowPlanDto.model_validate_json(dto.model_dump_json())

    plan = FlowPlanDtoMapper.to_flow_plan(received)

    assert plan == FlowPlan(
        plan_version=dto.plan_version,
        model_specs=ModelSpecs(
            model_variant_id=dto.model_specs.model_variant_id,
            inputs=(
                InputSpec(name="input", shape=shape, dtype="float32"),
                InputSpec(name="mask", shape=(1,), dtype="bool"),
            ),
        ),
        ingress_replica_id=dto.ingress_replica_id,
        sink_replica_id=dto.sink_replica_id,
    )


@pytest.mark.parametrize("endpoint", ["ingress_replica_id", "sink_replica_id"])
def test_mapping_rejects_inconsistent_endpoint_variants(endpoint: str) -> None:
    dto = _flow_plan_dto(())
    other_variant = ModelVariantId(ModelId(model_name="other"), "fp32")
    endpoint_id = (
        PlannedIngressReplicaId(other_variant, replica_idx=0)
        if endpoint == "ingress_replica_id"
        else PlannedSinkReplicaId(other_variant, replica_idx=0)
    )
    invalid_dto = dto.model_copy(update={endpoint: endpoint_id})

    with pytest.raises(ValueError, match="Model variant id must be consistent"):
        FlowPlanDtoMapper.to_flow_plan(invalid_dto)


def test_mapping_rejects_duplicate_input_names() -> None:
    input_spec = InputSpecDto(name="input", shape=(1,), dtype="float32")
    dto = _flow_plan_dto((input_spec, input_spec))

    with pytest.raises(ValueError, match="Duplicate input names"):
        FlowPlanDtoMapper.to_flow_plan(dto)


@pytest.mark.parametrize(
    ("shape", "dtype", "error_type"),
    [
        ((-1,), "float32", ValueError),
        (("",), "float32", ValueError),
        ((1,), "invalid-dtype", TypeError),
    ],
)
def test_mapping_applies_model_spec_domain_validation(
    shape: tuple[int | str | None, ...],
    dtype: str,
    error_type: type[Exception],
) -> None:
    dto = _flow_plan_dto((InputSpecDto(name="input", shape=shape, dtype=dtype),))

    with pytest.raises(error_type):
        FlowPlanDtoMapper.to_flow_plan(dto)
