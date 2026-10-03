import itertools
import re
from typing import override

import numpy as np
import onnx
import onnx_tool
import sympy
from onnx.shape_inference import infer_shapes

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.adapters.outbound.model_profiler import (
    onnx_tool_custom_nodes as _onnx_tool_custom_nodes,  # noqa: F401
)
from model_manager.application.ports.outbound.shape_profile_computer import (
    ShapeProfileComputer,
)
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import ModelVariantInfo
from model_manager.domain.model_variant_topology import ModelVariantTopology
from model_manager.domain.shape_profile import (
    LayerShapeProperty,
    ShapePoint,
    ShapeProfile,
    TensorShapeProperty,
)
from shared.model.keys import TensorKey


class OnnxShapeProfileComputer(ShapeProfileComputer):
    @override
    def compute_shape_profiles(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
    ) -> dict[ShapePoint, ShapeProfile]:

        model_proto = self._load_and_infer_model(artifact_workspace)
        shape_points = self._compute_shape_points(model_variant_info)

        return {
            shape_point: self._compute_shape_profile(
                model_proto,
                topology,
                shape_point,
            )
            for shape_point in shape_points
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
    def _compute_shape_points(
        model_variant_info: ModelVariantInfo,
    ) -> list[ShapePoint]:
        static_dimensions = [
            (shape.name, shape.value) for shape in model_variant_info.static_shapes
        ]
        dynamic_options = [
            [
                (shape.name, value)
                for value in range(
                    shape.min_value,
                    shape.max_value + 1,
                    shape.step_size,
                )
            ]
            for shape in model_variant_info.dynamic_shapes
        ]

        if not dynamic_options:
            return [ShapePoint(dims=tuple(sorted(static_dimensions)))]

        return [
            ShapePoint(dims=tuple(sorted([*static_dimensions, *dynamic_dimensions])))
            for dynamic_dimensions in itertools.product(*dynamic_options)
        ]

    @staticmethod
    def _compute_shape_profile(
        model_proto: onnx.ModelProto,
        topology: ModelVariantTopology,
        shape_point: ShapePoint,
    ) -> ShapeProfile:
        tool_model = onnx_tool.Model(model_proto)
        if not tool_model.valid:
            raise ValueError("onnx-tool could not load the ONNX model")

        initializer_names = {
            initializer.name for initializer in model_proto.graph.initializer
        }
        tool_inputs = {
            value_info.name: OnnxShapeProfileComputer._create_input(
                value_info,
                shape_point,
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
                layer_properties[layer_key] = LayerShapeProperty(flops=0)
                continue

            node = tool_graph.nodemap.get(layer_key)
            if node is None:
                raise ValueError(f"Layer {layer_key!r} is missing from onnx-tool")

            layer_properties[layer_key] = LayerShapeProperty(
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

            tensor_properties[tensor_key] = TensorShapeProperty(
                shape=tuple(int(dimension) for dimension in tensor_shape)
            )

        return ShapeProfile(
            layer_properties=layer_properties,
            tensor_properties=tensor_properties,
        )

    @staticmethod
    def _create_input(
        value_info_proto: onnx.ValueInfoProto,
        shape_point: ShapePoint,
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

        shape = tuple(
            OnnxShapeProfileComputer._resolve_dimension(dimension, shape_point)
            for dimension in tensor_type.shape.dim
        )
        data_type = np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        )

        return np.zeros(shape, dtype=data_type)

    @staticmethod
    def _resolve_dimension(
        dimension: onnx.TensorShapeProto.Dimension,
        shape_point: ShapePoint,
    ) -> int:
        dimension_type = dimension.WhichOneof("value")
        if dimension_type == "dim_value":
            return int(dimension.dim_value)
        if dimension_type != "dim_param":
            raise ValueError("Anonymous dynamic dimension cannot be resolved")

        expression_text = dimension.dim_param
        if re.fullmatch(r"[A-Za-z0-9_+\-*/%(),\s]+", expression_text) is None:
            raise ValueError(f"Unsupported symbolic dimension: {expression_text!r}")

        bindings = dict(shape_point.dims)
        functions = {
            "ceil": sympy.ceiling,
            "ceiling": sympy.ceiling,
            "floor": sympy.floor,
            "Max": sympy.Max,
            "Min": sympy.Min,
        }
        identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression_text))
        unknown_identifiers = identifiers - bindings.keys() - functions.keys()
        if unknown_identifiers:
            raise ValueError(
                f"Unresolved symbolic dimension {expression_text!r}: "
                f"unknown identifiers {sorted(unknown_identifiers)!r}"
            )

        symbols = {name: sympy.Symbol(name) for name in bindings}
        expression = sympy.sympify(expression_text, locals={**symbols, **functions})
        substitutions = {symbols[name]: value for name, value in bindings.items()}
        resolved = expression.subs(substitutions)

        if resolved.free_symbols:
            raise ValueError(f"Unresolved symbolic dimension: {dimension.dim_param!r}")
        if resolved.is_integer is not True:
            raise ValueError(f"Non-integer symbolic dimension: {dimension.dim_param!r}")

        value = int(resolved)
        if value < 0:
            raise ValueError(f"Negative tensor dimension: {value}")
        return value
