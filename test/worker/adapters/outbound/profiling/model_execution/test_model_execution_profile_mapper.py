from integration.dto.worker_profiles.execution_profile_dto import (
    BackendDto,
    ModelExecutionProfileDto,
)
from shared.model.model import ModelId
from shared.model.model_variant import ModelVariantId
from shared.service.service import ServiceId
from shared.user.user import UserId
from worker.adapters.outbound.profiling.model_execution.mappers.model_execution_profile_mapper import (
    ModelExecutionProfileMapper,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    Backend,
    BackendExecutionProfile,
    LayerExecutionProfile,
    ModelExecutionProfile,
    ShapeExecutionProfile,
)
from worker.domain.profiling.model_execution.shape_point import ShapePoint


def test_round_trips_backend_execution_profiles() -> None:
    profile = ModelExecutionProfile(
        model_version_id=ModelVariantId(
            model_id=ModelId(owner_id=UserId(), model_name="model"),
            variant_tag="variant",
        ),
        shape_execution_profiles=(
            ShapeExecutionProfile(
                shape_point=ShapePoint(input_shape_points={"input": (1, 3)}),
                backend_profiles=(
                    BackendExecutionProfile(
                        backend=Backend.CPU,
                        layer_profiles={
                            "layer": LayerExecutionProfile(
                                execution_time_s=0.25,
                                memory_bytes=1024,
                            )
                        },
                    ),
                    BackendExecutionProfile(
                        backend=Backend.GPU,
                        layer_profiles={
                            "layer": LayerExecutionProfile(
                                execution_time_s=0.05,
                                memory_bytes=None,
                            )
                        },
                    ),
                ),
            ),
        ),
    )

    dto = ModelExecutionProfileMapper.model_execution_profile_to_dto(
        ServiceId(), profile
    )

    assert tuple(
        item.backend for item in dto.shape_execution_profiles[0].backend_profiles
    ) == (
        BackendDto.CPU,
        BackendDto.GPU,
    )
    stored_dto = ModelExecutionProfileDto.model_validate_json(dto.model_dump_json())
    assert (
        ModelExecutionProfileMapper.model_execution_profile_from_dto(stored_dto)
        == profile
    )
