import itertools
from typing import override

import numpy as np
import onnx
import onnx_tool
from onnx.shape_inference import infer_shapes

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.application.ports.outbound.max_cut_size_computer import (
    MaxCutSizeComputer,
)
from model_manager.application.ports.outbound.shape_profile_computer import (
    ShapeProfileComputer,
)
from model_manager.domain.input_shape_invariable_properties import (
    InputShapeInvariableTensorProperties,
)
from model_manager.domain.input_shape_variable_properties import (
    InputShapeVariableLayerProperties,
    InputShapeVariableModelProperties,
    InputShapeVariableProperties,
    InputShapeVariableTensorProperties,
    ModelInputConcreteShapes,
)
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import ModelVariantInfo
from model_manager.domain.model_variant_topology import ModelVariantTopology
from shared.model.keys import TensorKey


class OnnxShapeProfileComputer(ShapeProfileComputer):
    def __init__(self, max_cut_size_computer: MaxCutSizeComputer) -> None:
        super().__init__()
        self._max_cut_size_computer = max_cut_size_computer

    @override
    def compute_shape_profiles(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
        invariable_tensor_properties: dict[
            TensorKey, InputShapeInvariableTensorProperties
        ],
    ) -> dict[ModelInputConcreteShapes, InputShapeVariableProperties]:

        model_proto = self._load_and_infer_model(artifact_workspace)
        input_shape_configurations = self._compute_input_shape_configurations(
            model_variant_info
        )

        return {
            input_shapes: self._compute_shape_profile(
                model_proto,
                topology,
                input_shapes,
                invariable_tensor_properties,
            )
            for input_shapes in input_shape_configurations
        }

    @staticmethod
    def _load_and_infer_model(
        artifact_workspace: ArtifactWorkspace,
    ) -> onnx.ModelProto:

        if not artifact_workspace.entrypoint_path:
            raise ValueError("Entrypoint path must be set when extracting the model")
        model_proto = onnx.load_model(
            artifact_workspace.entrypoint_path.as_posix(), load_external_data=True
        )

        inferred_model = infer_shapes(model_proto)
        return inferred_model

    @staticmethod
    def _compute_input_shape_configurations(
        model_variant_info: ModelVariantInfo,
    ) -> list[ModelInputConcreteShapes]:
        dynamic_dimension_names = tuple(
            sorted(model_variant_info.dynamic_dimension_values)
        )
        dynamic_dimension_options = tuple(
            sorted(model_variant_info.dynamic_dimension_values[dimension_name].values)
            for dimension_name in dynamic_dimension_names
        )

        input_shape_configurations = []
        for dynamic_values in itertools.product(*dynamic_dimension_options):
            dimension_bindings = dict(
                zip(dynamic_dimension_names, dynamic_values, strict=True)
            )
            input_shapes = {
                input_name: tuple(
                    dimension_bindings[dimension]
                    if isinstance(dimension, str)
                    else dimension
                    for dimension in input_info.shape
                )
                for input_name, input_info in sorted(
                    model_variant_info.inputs_info.items()
                )
            }
            input_shape_configurations.append(
                ModelInputConcreteShapes(input_shapes=input_shapes)
            )

        return input_shape_configurations

    def _compute_shape_profile(
        self,
        model_proto: onnx.ModelProto,
        topology: ModelVariantTopology,
        input_shapes: ModelInputConcreteShapes,
        invariable_tensor_properties: dict[
            TensorKey, InputShapeInvariableTensorProperties
        ],
    ) -> InputShapeVariableProperties:
        tool_model = onnx_tool.Model(model_proto)
        if not tool_model.valid:
            raise ValueError("onnx-tool could not load the ONNX model")

        print(f"Computing shape profile for {input_shapes}")

        initializer_names = {
            initializer.name for initializer in model_proto.graph.initializer
        }
        tool_inputs = {
            value_info.name: OnnxShapeProfileComputer._create_input(
                value_info,
                input_shapes.input_shapes[value_info.name],
            )
            for value_info in model_proto.graph.input
            if value_info.name not in initializer_names
        }

        tool_graph = tool_model.graph
        tool_graph.shape_infer(tool_inputs)
        tool_graph.profile()

        layer_properties = {}
        for layer_key, layer_description in topology.layers().items():
            if layer_description.is_input or layer_description.is_output:
                layer_properties[layer_key] = InputShapeVariableLayerProperties(flops=0)
                continue

            node = tool_graph.nodemap.get(layer_key)
            if node is None:
                raise ValueError(f"Layer {layer_key!r} is missing from onnx-tool")

            layer_properties[layer_key] = InputShapeVariableLayerProperties(
                flops=float(2 * node.macs[0])
            )

        tensor_keys: set[TensorKey] = set()
        for layer_key in topology.layers():
            for edge_description in topology.get_layer_out_edges(layer_key).values():
                tensor_keys.update(edge_description.tensors)

        tensor_properties = {}
        for tensor_key in tensor_keys:
            tensor = tool_graph.tensormap.get(tensor_key)
            if tensor is None:
                raise ValueError(f"Tensor {tensor_key!r} is missing from onnx-tool")

            tensor_shape = tensor.get_shape()
            if any(
                not isinstance(dimension, (int, np.integer))
                for dimension in tensor_shape
            ):
                raise ValueError(
                    f"Tensor {tensor_key!r} has unresolved shape {tensor_shape!r}"
                )

            tensor_properties[tensor_key] = InputShapeVariableTensorProperties(
                shape=tuple(int(dimension) for dimension in tensor_shape)
            )

        max_cut_size_mib = (
            self._max_cut_size_computer.compute_model_variant_max_cut_size(
                topology, invariable_tensor_properties, tensor_properties
            )
        )

        return InputShapeVariableProperties(
            layer_properties=layer_properties,
            tensor_properties=tensor_properties,
            model_properties=InputShapeVariableModelProperties(
                max_cut_size_mib=max_cut_size_mib
            ),
        )

    @staticmethod
    def _create_input(
        value_info_proto: onnx.ValueInfoProto,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        if value_info_proto.type.WhichOneof("value") != "tensor_type":
            raise ValueError(
                f"Input {value_info_proto.name!r} does not describe a tensor"
            )

        tensor_type = value_info_proto.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
            raise ValueError(
                f"Input {value_info_proto.name!r} has an undefined data type"
            )

        data_type = np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        )

        return np.zeros(shape, dtype=data_type)
