from integration.dto.model.profiled_model_variant_dto import (
    ModelInputConcreteShapesDto,
    ProfiledModelVariantDto,
)
from shared.model.keys import LayerKey
from worker.domain.profiling.model_execution.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)


class ModelStaticProfileMapper:
    @staticmethod
    def model_static_profile_from_dto(
        dto: ProfiledModelVariantDto,
    ) -> ModelStaticProfile:

        model_variant_id = dto.model_variant_id
        artifact_ref = dto.artifact_ref

        input_shapes_dtos: list[ModelInputConcreteShapesDto] = (
            dto.profile.input_shape_configurations
        )

        input_shape_configurations = [
            ModelInputConcreteShapes(input_shapes=input_shapes_dto.input_shapes)
            for input_shapes_dto in input_shapes_dtos
        ]

        contractions: list[tuple[LayerKey, ...]] = dto.profile.optimization_contractions

        return ModelStaticProfile(
            model_variant_id=model_variant_id,
            artifact_ref=artifact_ref,
            input_shape_configurations=tuple(input_shape_configurations),
            contractions=tuple(contractions),
        )
