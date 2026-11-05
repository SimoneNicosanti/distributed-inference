from integration.dto.model.model_variant_dto import (
    DynamicShapeValuesDto,
    InputInfoDto,
    ModelVariantInfoDto,
)
from integration.dto.model.profiled_model_variant_dto import (
    EdgeDescriptionDto,
    InputShapeInvariableLayerPropertiesDto,
    InputShapeInvariablePropertiesDto,
    InputShapeInvariableTensorPropertiesDto,
    InputShapeVariableLayerPropertiesDto,
    InputShapeVariableModelPropertiesDto,
    InputShapeVariablePropertiesDto,
    InputShapeVariableTensorPropertiesDto,
    LayerDescriptionDto,
    ModelInputConcreteShapesDto,
    ProfiledModelVariantDto,
    ProfileDto,
    TopologyDto,
)
from model_manager.domain.input_shape_invariable_properties import (
    InputShapeInvariableLayerProperties,
    InputShapeInvariableProperties,
    InputShapeInvariableTensorProperties,
)
from model_manager.domain.input_shape_variable_properties import (
    InputShapeVariableLayerProperties,
    InputShapeVariableModelProperties,
    InputShapeVariableProperties,
    InputShapeVariableTensorProperties,
    ModelInputConcreteShapes,
)
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

        variable_properties = [
            InputShapeVariablePropertiesDto(
                input_shapes=ModelInputConcreteShapesDto(
                    input_shapes=dict(input_shapes.input_shapes)
                ),
                layer_properties={
                    layer_key: InputShapeVariableLayerPropertiesDto(
                        flops=property_.flops
                    )
                    for layer_key, property_ in properties.layer_properties.items()
                },
                tensor_properties={
                    tensor_key: InputShapeVariableTensorPropertiesDto(
                        shape=property_.shape
                    )
                    for tensor_key, property_ in properties.tensor_properties.items()
                },
                model_properties=InputShapeVariableModelPropertiesDto(
                    max_cut_size_mib=properties.model_properties.max_cut_size_mib,
                ),
            )
            for input_shapes, properties in profile.variable_properties.items()
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
                invariable_properties=InputShapeInvariablePropertiesDto(
                    layer_properties={
                        layer_key: InputShapeInvariableLayerPropertiesDto(
                            weight_properties={
                                tensor_key: InputShapeInvariableTensorPropertiesDto(
                                    data_type=weight.data_type,
                                    shape_expression=weight.shape_expression,
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
                            profile.invariable_properties.layer_properties.items()
                        )
                    },
                    tensor_properties={
                        tensor_key: InputShapeInvariableTensorPropertiesDto(
                            data_type=property_.data_type,
                            data_type_size_in_bytes=property_.data_type_size_in_bytes,
                            shape_expression=property_.shape_expression,
                        )
                        for tensor_key, property_ in (
                            profile.invariable_properties.tensor_properties.items()
                        )
                    },
                ),
                variable_properties=variable_properties,
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

        variable_properties: dict[
            ModelInputConcreteShapes, InputShapeVariableProperties
        ] = {}
        for properties_dto in dto.profile.variable_properties:
            input_shapes = ModelInputConcreteShapes(
                input_shapes=properties_dto.input_shapes.input_shapes,
            )
            if input_shapes in variable_properties:
                raise ValueError(f"Duplicate input shapes in Redis DTO: {input_shapes}")

            variable_properties[input_shapes] = InputShapeVariableProperties(
                layer_properties={
                    layer_key: InputShapeVariableLayerProperties(flops=property_.flops)
                    for layer_key, property_ in (
                        properties_dto.layer_properties.items()
                    )
                },
                tensor_properties={
                    tensor_key: InputShapeVariableTensorProperties(
                        shape=property_.shape
                    )
                    for tensor_key, property_ in (
                        properties_dto.tensor_properties.items()
                    )
                },
                model_properties=InputShapeVariableModelProperties(
                    max_cut_size_mib=properties_dto.model_properties.max_cut_size_mib,
                ),
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
                invariable_properties=InputShapeInvariableProperties(
                    layer_properties={
                        layer_key: InputShapeInvariableLayerProperties(
                            weight_properties={
                                tensor_key: InputShapeInvariableTensorProperties(
                                    data_type=weight.data_type,
                                    shape_expression=weight.shape_expression,
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
                            dto.profile.invariable_properties.layer_properties.items()
                        )
                    },
                    tensor_properties={
                        tensor_key: InputShapeInvariableTensorProperties(
                            data_type=property_.data_type,
                            data_type_size_in_bytes=property_.data_type_size_in_bytes,
                            shape_expression=property_.shape_expression,
                        )
                        for tensor_key, property_ in (
                            dto.profile.invariable_properties.tensor_properties.items()
                        )
                    },
                ),
                variable_properties=variable_properties,
                optimization_contractions=tuple(dto.profile.optimization_contractions),
            ),
            artifact_ref=dto.artifact_ref,
        )
