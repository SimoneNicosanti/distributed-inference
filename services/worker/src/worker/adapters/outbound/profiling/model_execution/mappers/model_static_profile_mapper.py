from integration.dto.model.profiled_model_variant_dto import (
    ProfiledModelVariantDto,
    ShapePointDto,
)
from shared.model.keys import LayerKey
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)
from worker.domain.profiling.model_execution.shape_point import ShapePoint


class ModelStaticProfileMapper:
    @staticmethod
    def model_static_profile_from_dto(
        dto: ProfiledModelVariantDto,
    ) -> ModelStaticProfile:

        model_variant_id = dto.model_variant_id
        artifact_ref = dto.artifact_ref

        shape_point_dtos: list[ShapePointDto] = dto.profile.shape_points

        shape_points = []
        for shape_point_dto in shape_point_dtos:
            input_shape_points = {
                input_shape_point.name: input_shape_point.shape
                for input_shape_point in shape_point_dto.input_shape_points
            }

            shape_points.append(ShapePoint(input_shape_points=input_shape_points))

        contractions: list[tuple[LayerKey, ...]] = dto.profile.optimization_contractions

        return ModelStaticProfile(
            model_variant_id=model_variant_id,
            artifact_ref=artifact_ref,
            shape_points=tuple(shape_points),
            contractions=tuple(contractions),
        )
