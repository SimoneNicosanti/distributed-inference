from controller.adapters.outbound.profile.mappers.model_variant_execution_profile_mapper import (
    ModelVariantExecutionProfileMapper,
)
from controller.domain.profile.model_variant_execution_profile import Backend
from integration.dto.worker_profiles.execution_profile_dto import (
    BackendDto,
    BackendExecutionProfileDto,
    LayerExecutionProfileDto,
    ModelExecutionProfileDto,
    ShapeExecutionProfileDto,
    ShapePointDto,
)
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId
from shared.user.user import UserId


def test_maps_all_backend_profiles() -> None:
    layer = LayerExecutionProfileDto(
        layer_key="layer",
        execution_time_s=0.25,
        memory_bytes=1024,
    )
    dto = ModelExecutionProfileDto(
        worker_id=ServiceId(),
        model_variant_id=ModelVariantId(
            model_id=ModelId(owner_id=UserId(), model_name="model"),
            variant_tag="variant",
        ),
        shape_execution_profiles=(
            ShapeExecutionProfileDto(
                shape_point=ShapePointDto(input_shape_points={"input": (1, 3)}),
                backend_profiles=(
                    BackendExecutionProfileDto(
                        backend=BackendDto.CPU,
                        layer_profiles={"layer": layer},
                    ),
                    BackendExecutionProfileDto(
                        backend=BackendDto.GPU,
                        layer_profiles={"layer": layer},
                    ),
                ),
            ),
        ),
    )

    profile = (
        ModelVariantExecutionProfileMapper.model_variant_execution_profile_from_dto(dto)
    )

    shape_profile = profile.shape_execution_profiles[0]
    assert tuple(item.backend for item in shape_profile.backend_profiles) == (
        Backend.CPU,
        Backend.GPU,
    )
    assert (
        shape_profile.backend_profiles[0].layer_profiles["layer"].memory_bytes == 1024
    )
