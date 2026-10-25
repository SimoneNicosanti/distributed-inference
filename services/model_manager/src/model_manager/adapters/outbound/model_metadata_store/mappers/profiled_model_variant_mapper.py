from integration.dto.model.model_variant_dto import (
    DynamicShapeValuesDto,
    InputInfoDto,
    ModelVariantInfoDto,
)
from integration.dto.model.profiled_model_variant_dto import (
    EdgeDescriptionDto,
    InputShapePointDto,
    LayerDescriptionDto,
    LayerPropertyDto,
    LayerShapePropertyDto,
    ProfiledModelVariantDto,
    ProfileDto,
    ShapePointDto,
    ShapeProfileDto,
    TensorPropertyDto,
    TensorShapePropertyDto,
    TopologyDto,
    WeightPropertyDto,
)
from model_manager.domain.layer_property import LayerProperty, WeightProperty
from model_manager.domain.model_variant import (
    DynamicShapeValues,
    InputInfo,
    ModelVariant,
    ModelVariantInfo,
)
from model_manager.domain.model_variant_profile import ModelVariantProfile
from model_manager.domain.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
)
from model_manager.domain.profiled_model_variant import ProfiledModelVariant
from model_manager.domain.shape_profile import (
    InputShapePoint,
    LayerShapeProperty,
    ShapePoint,
    ShapeProfile,
    TensorShapeProperty,
)
from model_manager.domain.tensor_property import TensorProperty


class ProfiledModelVariantMapper:
    @staticmethod
    def to_profiled_model_variant_dto(
        profiled_model_variant: ProfiledModelVariant,
    ) -> ProfiledModelVariantDto:
        profile = profiled_model_variant.profile

        topology = TopologyDto(
            layers={
                layer_key: LayerDescriptionDto(
                    name=description.name,
                    type=description.type,
                    is_input=description.is_input,
                    is_output=description.is_output,
                )
                for layer_key, description in profile.topology.layers().items()
            },
            edges=[
                EdgeDescriptionDto(
                    source=description.source,
                    target=description.target,
                    tensors=set(description.tensors),
                )
                for description in profile.topology.edges().values()
            ],
        )

        shape_profiles = [
            ShapeProfileDto(
                shape_point=ShapePointDto(
                    input_shape_points=tuple(
                        InputShapePointDto(name=point.name, shape=point.shape)
                        for point in shape_point.input_shape_points
                    )
                ),
                layer_properties={
                    layer_key: LayerShapePropertyDto(flops=property_.flops)
                    for layer_key, property_ in shape_profile.layer_properties.items()
                },
                tensor_properties={
                    tensor_key: TensorShapePropertyDto(shape=property_.shape)
                    for tensor_key, property_ in shape_profile.tensor_properties.items()
                },
            )
            for shape_point, shape_profile in profile.shape_profiles.items()
        ]

        return ProfiledModelVariantDto(
            model_variant_id=profiled_model_variant.model_variant_id,
            info=ModelVariantInfoDto(
                precision=(
                    profiled_model_variant.model_variant.model_variant_info.precision
                ),
                quantization=(
                    profiled_model_variant.model_variant.model_variant_info.quantization
                ),
                accuracies=list(
                    profiled_model_variant.model_variant.model_variant_info.accuracies
                ),
                format=profiled_model_variant.model_variant.model_variant_info.format,
                inputs_info={
                    name: InputInfoDto(shape=info.shape)
                    for name, info in profiled_model_variant.model_variant.model_variant_info.inputs_info.items()
                },
                dynamic_dimension_values={
                    name: DynamicShapeValuesDto(values=values.values)
                    for name, values in profiled_model_variant.model_variant.model_variant_info.dynamic_dimension_values.items()
                },
            ),
            profile=ProfileDto(
                topology=topology,
                invariant_layer_properties={
                    layer_key: LayerPropertyDto(
                        weight_properties={
                            tensor_key: WeightPropertyDto(
                                data_type=weight.data_type,
                                shape=weight.shape,
                                data_type_size_in_bytes=(
                                    weight.data_type_size_in_bytes
                                ),
                            )
                            for tensor_key, weight in (
                                property_.weight_properties.items()
                            )
                        }
                    )
                    for layer_key, property_ in (
                        profile.invariant_layer_properties.items()
                    )
                },
                invariant_tensor_properties={
                    tensor_key: TensorPropertyDto(
                        data_type=property_.data_type,
                        data_type_size_in_bytes=property_.data_type_size_in_bytes,
                        shape_expression=property_.shape_expression,
                    )
                    for tensor_key, property_ in (
                        profile.invariant_tensor_properties.items()
                    )
                },
                shape_profiles=shape_profiles,
                optimization_contractions=list(profile.optimization_contractions),
            ),
            artifact_ref=profiled_model_variant.artifact_ref,
        )

    @staticmethod
    def from_profiled_model_variant_dto(
        dto: ProfiledModelVariantDto,
    ) -> ProfiledModelVariant:
        topology = ModelVariantTopology()
        for layer_key, layer_description in dto.profile.topology.layers.items():
            topology.add_layer(
                layer_key,
                LayerDescription(
                    name=layer_description.name,
                    type=layer_description.type,
                    is_input=layer_description.is_input,
                    is_output=layer_description.is_output,
                ),
            )
        for edge_description in dto.profile.topology.edges:
            topology.add_edge(
                (edge_description.source, edge_description.target),
                EdgeDescription(
                    source=edge_description.source,
                    target=edge_description.target,
                    tensors=frozenset(edge_description.tensors),
                ),
            )

        shape_profiles: dict[ShapePoint, ShapeProfile] = {}
        for shape_profile_dto in dto.profile.shape_profiles:
            shape_point = ShapePoint(
                input_shape_points=tuple(
                    InputShapePoint(name=point.name, shape=point.shape)
                    for point in shape_profile_dto.shape_point.input_shape_points
                )
            )
            if shape_point in shape_profiles:
                raise ValueError(f"Duplicate shape point in Redis DTO: {shape_point}")

            shape_profiles[shape_point] = ShapeProfile(
                layer_properties={
                    layer_key: LayerShapeProperty(flops=property_.flops)
                    for layer_key, property_ in (
                        shape_profile_dto.layer_properties.items()
                    )
                },
                tensor_properties={
                    tensor_key: TensorShapeProperty(shape=property_.shape)
                    for tensor_key, property_ in (
                        shape_profile_dto.tensor_properties.items()
                    )
                },
            )

        return ProfiledModelVariant(
            model_variant=ModelVariant(
                model_variant_id=dto.model_variant_id,
                model_variant_info=ModelVariantInfo(
                    precision=dto.info.precision,
                    quantization=dto.info.quantization,
                    accuracies=tuple(dto.info.accuracies),
                    format=dto.info.format,
                    inputs_info={
                        name: InputInfo(shape=info.shape)
                        for name, info in dto.info.inputs_info.items()
                    },
                    dynamic_dimension_values={
                        name: DynamicShapeValues(values=values.values)
                        for name, values in (dto.info.dynamic_dimension_values.items())
                    },
                ),
            ),
            profile=ModelVariantProfile(
                topology=topology,
                invariant_layer_properties={
                    layer_key: LayerProperty(
                        weight_properties={
                            tensor_key: WeightProperty(
                                data_type=weight.data_type,
                                shape=weight.shape,
                                data_type_size_in_bytes=(
                                    weight.data_type_size_in_bytes
                                ),
                            )
                            for tensor_key, weight in (
                                property_.weight_properties.items()
                            )
                        }
                    )
                    for layer_key, property_ in (
                        dto.profile.invariant_layer_properties.items()
                    )
                },
                invariant_tensor_properties={
                    tensor_key: TensorProperty(
                        data_type=property_.data_type,
                        data_type_size_in_bytes=property_.data_type_size_in_bytes,
                        shape_expression=property_.shape_expression,
                    )
                    for tensor_key, property_ in (
                        dto.profile.invariant_tensor_properties.items()
                    )
                },
                shape_profiles=shape_profiles,
                optimization_contractions=tuple(dto.profile.optimization_contractions),
            ),
            artifact_ref=dto.artifact_ref,
        )
