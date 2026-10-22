from controller.domain.profile.model_variant_execution_profile import (
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
            layer_profiles: dict[str, LayerExecutionProfile] = {}
            for layer_key, layer_profile in shape_profile.layer_profiles.items():
                if layer_profile.layer_key != layer_key:
                    raise ValueError(
                        "Stored execution profile layer key does not match its index"
                    )
                layer_profiles[layer_key] = LayerExecutionProfile(
                    execution_time_s=(
                        layer_profile.cpu_execution_profile.execution_time
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
                    layer_profiles=layer_profiles,
                )
            )

        return ModelVariantExecutionProfile(
            shape_execution_profiles=tuple(shape_execution_profiles)
        )
