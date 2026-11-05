from controller.domain.profile.model_variant_static_profile import (
    LayerProperty,
    LayerShapeProperty,
    ModelInputConcreteShapes,
    ModelVariantInfo,
    ModelVariantStaticProfile,
    ShapeProfile,
    TensorProperty,
    TensorShapeProperty,
    WeightProperty,
)
from controller.domain.profile.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
)
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.profiled_model_variant_dto import (
    InputShapeInvariableTensorPropertiesDto,
    ProfiledModelVariantDto,
)


class ModelVariantStaticProfileMapper:
    @staticmethod
    def model_variant_static_profile_from_dto(
        dto: ProfiledModelVariantDto,
        model_dto: ModelDto,
    ) -> ModelVariantStaticProfile:
        profile = dto.profile

        return ModelVariantStaticProfile(
            model_variant_id=dto.model_variant_id,
            info=ModelVariantInfo(
                task=model_dto.model_task,
                accuracies=tuple(dto.info.accuracies),
            ),
            topology=ModelVariantTopology(
                layers={
                    layer_key: LayerDescription(
                        name=description.name,
                        type=description.type,
                        is_input=description.is_input,
                        is_output=description.is_output,
                    )
                    for layer_key, description in profile.topology.layers.items()
                },
                edges=(
                    EdgeDescription(
                        source=description.source,
                        target=description.target,
                        tensors=frozenset(description.tensors),
                    )
                    for description in profile.topology.edges
                ),
            ),
            invariant_layer_properties={
                layer_key: LayerProperty(
                    weight_properties={
                        tensor_key: ModelVariantStaticProfileMapper._weight_property_from_dto(
                            weight
                        )
                        for tensor_key, weight in property_.weight_properties.items()
                    }
                )
                for layer_key, property_ in profile.invariable_properties.layer_properties.items()
            },
            invariant_tensor_properties={
                tensor_key: TensorProperty(
                    data_type=property_.data_type,
                    data_type_size_in_bytes=property_.data_type_size_in_bytes,
                    shape_expression=property_.shape_expression,
                )
                for tensor_key, property_ in (
                    profile.invariable_properties.tensor_properties.items()
                )
            },
            shape_profiles=tuple(
                ShapeProfile(
                    max_cut_size_mib=shape_profile.model_properties.max_cut_size_mib,
                    input_shapes=ModelInputConcreteShapes(
                        input_shapes=shape_profile.input_shapes.input_shapes,
                    ),
                    layer_properties={
                        layer_key: LayerShapeProperty(flops=property_.flops)
                        for layer_key, property_ in (
                            shape_profile.layer_properties.items()
                        )
                    },
                    tensor_properties={
                        tensor_key: TensorShapeProperty(shape=property_.shape)
                        for tensor_key, property_ in (
                            shape_profile.tensor_properties.items()
                        )
                    },
                )
                for shape_profile in profile.variable_properties
            ),
            optimization_contractions=tuple(profile.optimization_contractions),
        )

    @staticmethod
    def _weight_property_from_dto(
        dto: InputShapeInvariableTensorPropertiesDto,
    ) -> WeightProperty:
        shape: list[int] = []
        for dimension in dto.shape_expression:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise ValueError("Weight dimensions must be concrete integers")
            shape.append(dimension)
        return WeightProperty(
            data_type=dto.data_type,
            shape=tuple(shape),
            data_type_size_in_bytes=dto.data_type_size_in_bytes,
        )
