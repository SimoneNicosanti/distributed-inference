from controller.domain.profile.model_variant_static_profile import (
    InputShapePoint,
    LayerProperty,
    LayerShapeProperty,
    ModelVariantInfo,
    ModelVariantStaticProfile,
    ShapePoint,
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
from integration.dto.model.profiled_model_variant_dto import ProfiledModelVariantDto


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
                        tensor_key: WeightProperty(
                            data_type=weight.data_type,
                            shape=weight.shape,
                            data_type_size_in_bytes=weight.data_type_size_in_bytes,
                        )
                        for tensor_key, weight in property_.weight_properties.items()
                    }
                )
                for layer_key, property_ in profile.invariant_layer_properties.items()
            },
            invariant_tensor_properties={
                tensor_key: TensorProperty(
                    data_type=property_.data_type,
                    data_type_size_in_bytes=property_.data_type_size_in_bytes,
                    shape_expression=property_.shape_expression,
                )
                for tensor_key, property_ in (
                    profile.invariant_tensor_properties.items()
                )
            },
            shape_profiles=tuple(
                ShapeProfile(
                    shape_point=ShapePoint(
                        input_shape_points=tuple(
                            InputShapePoint(name=point.name, shape=point.shape)
                            for point in shape_profile.shape_point.input_shape_points
                        )
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
                for shape_profile in profile.shape_profiles
            ),
            optimization_contractions=tuple(profile.optimization_contractions),
        )
