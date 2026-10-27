from controller.domain.profile.model_variant_execution_profile import (
    Backend,
    BackendExecutionProfile,
    LayerExecutionProfile,
    ModelVariantExecutionProfile,
    ShapeExecutionProfile,
)
from controller.domain.profile.model_variant_static_profile import (
    InputShapePoint,
    ShapePoint,
)
from integration.dto.worker_profiles.execution_profile_dto import (
    ModelExecutionProfileDto,
)


class ModelVariantExecutionProfileMapper:
    @staticmethod
    def model_variant_execution_profile_from_dto(
        dto: ModelExecutionProfileDto,
    ) -> ModelVariantExecutionProfile:
        shape_execution_profiles: list[ShapeExecutionProfile] = []

        for shape_profile in dto.shape_execution_profiles:
            backend_profiles: list[BackendExecutionProfile] = []
            for backend_profile in shape_profile.backend_profiles:
                layer_profiles: dict[str, LayerExecutionProfile] = {}
                for layer_key, layer_profile in backend_profile.layer_profiles.items():
                    if layer_profile.layer_key != layer_key:
                        raise ValueError(
                            "Stored execution profile layer key does not match its index"
                        )
                    layer_profiles[layer_key] = LayerExecutionProfile(
                        execution_time_s=layer_profile.execution_time_s,
                        memory_bytes=layer_profile.memory_bytes,
                    )
                backend_profiles.append(
                    BackendExecutionProfile(
                        backend=Backend(backend_profile.backend.value),
                        layer_profiles=layer_profiles,
                    )
                )

            shape_execution_profiles.append(
                ShapeExecutionProfile(
                    shape_point=ShapePoint(
                        input_shape_points=tuple(
                            InputShapePoint(name=name, shape=shape)
                            for name, shape in sorted(
                                shape_profile.shape_point.input_shape_points.items()
                            )
                        )
                    ),
                    backend_profiles=tuple(backend_profiles),
                )
            )

        return ModelVariantExecutionProfile(
            shape_execution_profiles=tuple(shape_execution_profiles)
        )
