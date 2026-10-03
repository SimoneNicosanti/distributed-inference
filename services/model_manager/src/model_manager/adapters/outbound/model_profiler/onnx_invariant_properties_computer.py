from typing import override

import numpy as np
import onnx
from onnx.shape_inference import infer_shapes

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.application.ports.outbound.invariant_properties_computer import (
    InvariantPropertiesComputer,
)
from model_manager.domain.layer_property import LayerProperty, WeightProperty
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import ModelVariantInfo
from model_manager.domain.model_variant_topology import ModelVariantTopology
from model_manager.domain.tensor_property import TensorProperty
from shared.model.keys import LayerKey, TensorKey


class OnnxInvariantPropertiesComputer(InvariantPropertiesComputer):
    @override
    def compute_layer_invariant_properties(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
    ) -> dict[LayerKey, LayerProperty]:

        invariant_properties: dict[LayerKey, LayerProperty] = {}

        model_proto = self._load_and_infer_model(artifact_workspace)

        initializers_dict: dict[str, onnx.TensorProto] = {
            tensor.name: tensor for tensor in model_proto.graph.initializer
        }

        node_proto_dict: dict[str, onnx.NodeProto] = {
            node.name: node for node in model_proto.graph.node
        }

        for layer_key, layer_description in topology.layers().items():
            weight_properties: dict[TensorKey, WeightProperty] = {}
            if not layer_description.is_input and not layer_description.is_output:
                node_proto = node_proto_dict[layer_key]

                for input_name in node_proto.input:
                    if input_name in initializers_dict:
                        weight_properties[input_name] = self._compute_weight_properties(
                            initializers_dict[input_name],
                        )
            invariant_properties[layer_key] = LayerProperty(
                weight_properties=weight_properties
            )

        return invariant_properties

    @staticmethod
    def _load_and_infer_model(
        artifact_workspace: ArtifactWorkspace,
    ) -> onnx.ModelProto:

        if not artifact_workspace.entrypoint_path:
            raise ValueError("Entrypoint path must be set when extracting the model")
        model_proto = onnx.load_model(
            artifact_workspace.entrypoint_path.as_posix(), load_external_data=False
        )

        inferred_model = infer_shapes(model_proto)
        return inferred_model

    @staticmethod
    def _compute_weight_properties(
        onnx_tensor_proto: onnx.TensorProto,
    ) -> WeightProperty:
        data_type = np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(onnx_tensor_proto.data_type)
        )
        shape = tuple(int(dimension) for dimension in onnx_tensor_proto.dims)
        data_type_size_in_bytes = int(data_type.itemsize)

        return WeightProperty(
            data_type=data_type.name,
            shape=shape,
            data_type_size_in_bytes=data_type_size_in_bytes,
        )

    @staticmethod
    def _compute_tensor_properties(
        onnx_tensor_proto: onnx.ValueInfoProto,
    ) -> TensorProperty:
        if onnx_tensor_proto.type.WhichOneof("value") != "tensor_type":
            raise ValueError(
                f"Value {onnx_tensor_proto.name!r} does not describe a tensor"
            )

        tensor_type = onnx_tensor_proto.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.UNDEFINED:
            raise ValueError(
                f"Tensor {onnx_tensor_proto.name!r} has an undefined data type"
            )

        data_type = np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        )

        shape_expression: list[int | str] = []
        for dimension in tensor_type.shape.dim:
            dimension_type = dimension.WhichOneof("value")
            if dimension_type == "dim_value":
                shape_expression.append(int(dimension.dim_value))
            elif dimension_type == "dim_param":
                shape_expression.append(dimension.dim_param)
            else:
                raise ValueError(
                    f"Tensor {onnx_tensor_proto.name!r} has an anonymous dimension"
                )

        return TensorProperty(
            data_type=data_type.name,
            data_type_size_in_bytes=int(data_type.itemsize),
            shape_expression=tuple(shape_expression),
        )

    @override
    def compute_tensor_invariant_properties(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
    ) -> dict[TensorKey, TensorProperty]:

        model_proto = self._load_and_infer_model(artifact_workspace)

        value_info_proto_dict: dict[str, onnx.ValueInfoProto] = {
            value_info.name: value_info
            for value_info in (
                *model_proto.graph.value_info,
                *model_proto.graph.input,
                *model_proto.graph.output,
            )
        }

        tensor_keys: set[TensorKey] = set()
        for layer_key in topology.layers():
            for edge_description in topology.get_layer_out_edges(layer_key).values():
                tensor_keys.update(edge_description.tensors)

        invariant_properties: dict[TensorKey, TensorProperty] = {}
        for tensor_key in tensor_keys:
            value_info_proto = value_info_proto_dict.get(tensor_key)
            if value_info_proto is None:
                raise ValueError(
                    f"Tensor {tensor_key!r} has no inferred ONNX value information"
                )

            invariant_properties[tensor_key] = self._compute_tensor_properties(
                value_info_proto
            )

        return invariant_properties
