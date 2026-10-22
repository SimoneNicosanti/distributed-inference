from integration.dto.worker_profiles.execution_profile_dto import (
    BackendLayerExecutionProfileDto,
    LayerExecutionProfileDto,
    ModelExecutionProfileDto,
    ShapeExecutionProfileDto,
    ShapePointDto,
)
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId
from worker.domain.profiling.model_execution.model_execution_profile import (
    BackendLayerExecutionProfile,
    LayerExecutionProfile,
    ModelExecutionProfile,
    ShapeExecutionProfile,
)
from worker.domain.profiling.model_execution.shape_point import ShapePoint


class ModelExecutionProfileMapper:
    @staticmethod
    def model_execution_profile_to_dto(
        worker_id: WorkerId,
        profile: ModelExecutionProfile,
    ) -> ModelExecutionProfileDto:
        return ModelExecutionProfileDto(
            worker_id=worker_id,
            model_variant_id=profile.model_version_id,
            shape_execution_profiles=tuple(
                ShapeExecutionProfileDto(
                    shape_point=ShapePointDto(
                        input_shape_points=dict(
                            shape_profile.shape_point.input_shape_points
                        )
                    ),
                    layer_profiles={
                        layer_key: LayerExecutionProfileDto(
                            layer_key=layer_profile.layer_key,
                            cpu_execution_profile=BackendLayerExecutionProfileDto(
                                execution_time=(
                                    layer_profile.cpu_execution_profile.execution_time
                                ),
                                memory=layer_profile.cpu_execution_profile.memory,
                            ),
                            gpu_execution_profile=(
                                None
                                if layer_profile.gpu_execution_profile is None
                                else BackendLayerExecutionProfileDto(
                                    execution_time=(
                                        layer_profile.gpu_execution_profile.execution_time
                                    ),
                                    memory=layer_profile.gpu_execution_profile.memory,
                                )
                            ),
                        )
                        for layer_key, layer_profile in shape_profile.layer_profiles.items()
                    },
                )
                for shape_profile in profile.shape_execution_profiles
            ),
        )

    @staticmethod
    def model_execution_profile_from_dto(
        dto: ModelExecutionProfileDto,
    ) -> ModelExecutionProfile:
        return ModelExecutionProfile(
            model_version_id=dto.model_variant_id,
            shape_execution_profiles=tuple(
                ShapeExecutionProfile(
                    shape_point=ShapePoint(
                        input_shape_points=shape_profile.shape_point.input_shape_points
                    ),
                    layer_profiles={
                        layer_key: LayerExecutionProfile(
                            layer_key=layer_profile.layer_key,
                            cpu_execution_profile=BackendLayerExecutionProfile(
                                execution_time=(
                                    layer_profile.cpu_execution_profile.execution_time
                                ),
                                memory=layer_profile.cpu_execution_profile.memory,
                            ),
                            gpu_execution_profile=(
                                None
                                if layer_profile.gpu_execution_profile is None
                                else BackendLayerExecutionProfile(
                                    execution_time=(
                                        layer_profile.gpu_execution_profile.execution_time
                                    ),
                                    memory=layer_profile.gpu_execution_profile.memory,
                                )
                            ),
                        )
                        for layer_key, layer_profile in shape_profile.layer_profiles.items()
                    },
                )
                for shape_profile in dto.shape_execution_profiles
            ),
        )

    @staticmethod
    def validate_dto_identity(
        dto: ModelExecutionProfileDto,
        worker_id: WorkerId,
        model_variant_id: ModelVariantId | None = None,
    ) -> None:
        if dto.worker_id != worker_id:
            raise ValueError(
                "Stored execution profile worker does not match its Redis key"
            )
        if model_variant_id is not None and dto.model_variant_id != model_variant_id:
            raise ValueError(
                "Stored execution profile variant does not match its Redis field"
            )
