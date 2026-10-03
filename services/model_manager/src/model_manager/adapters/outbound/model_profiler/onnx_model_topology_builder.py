from typing import override

import onnx

from artifacts.contracts.artifact_workspace import (
    ArtifactWorkspace,
)
from model_manager.application.ports.outbound.model_topology_builder import (
    ModelTopologyBuilder,
)
from model_manager.domain import model_variant_topology
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import (
    ModelVariantInfo,
)
from model_manager.domain.model_variant_topology import (
    EdgeDescription,
    LayerDescription,
    ModelVariantTopology,
)


class OnnxModelTopologyBuilder(ModelTopologyBuilder):
    @override
    def build_model_topology(
        self,
        paths: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
    ) -> ModelVariantTopology:

        if paths.entrypoint_path is None:
            raise ValueError(
                "Entrypoint path must be set when extracting the model graph"
            )
        path = paths.entrypoint_path

        model_proto = onnx.load_model(path.as_posix(), load_external_data=False)

        inferred_model = OnnxModelTopologyBuilder.__infer_model_shape(model_proto)
        topology: ModelVariantTopology = ModelVariantTopology()

        try:
            onnx.checker.check_model(path.as_posix(), full_check=True)
            # model_proto = infer_shapes(inferred_model)

            OnnxModelTopologyBuilder._init_model_topology(topology)
            OnnxModelTopologyBuilder._add_topology_nodes(inferred_model, topology)
            OnnxModelTopologyBuilder._add_topology_edges(inferred_model, topology)
            OnnxModelTopologyBuilder._clear_topology(topology)

        except onnx.checker.ValidationError as e:
            raise Exception("Invalid ONNX model: " + str(e))

        return topology

    @staticmethod
    def __infer_model_shape(model_proto: onnx.ModelProto) -> onnx.ModelProto:
        ## Since we have optimize using onnxruntime, we need to use ort infer shape method
        ## There might be non supported op in the model for which ort infer shape silently fails
        from onnx.shape_inference import infer_shapes

        inferred_model = infer_shapes(model_proto, data_prop=True)

        # inferred_model = SymbolicShapeInference.infer_shapes(
        #     model_proto,
        #     int_max=2**31 - 1,
        #     auto_merge=True,
        #     guess_output_rank=True,  ## TODO RICONTROLLA
        #     verbose=3,
        # )
        # if inferred_model is None:
        #     raise Exception("Failed to infer model shape")

        return inferred_model

    @staticmethod
    def _clear_topology(model_graph: ModelVariantTopology) -> None:
        # Removing all nodes that are not reachable from the input node
        # In some cases, there are nodes used to define the weights
        # Or pre-processing operations on them (like quantization)
        reachable = model_graph.get_reachable_from_layer(
            model_variant_topology.INPUT_LAYER_NAME
        ).keys()

        all_layers = model_graph.layers().keys()

        unreachable = set(all_layers) - set(reachable)

        if model_variant_topology.OUTPUT_LAYER_NAME in unreachable:
            raise Exception("The model output is unreachable")

        model_graph.remove_layers_from_iterable(unreachable)

    @staticmethod
    def _init_model_topology(
        topology: ModelVariantTopology,
    ) -> None:

        input_layer_description = LayerDescription(
            name=model_variant_topology.INPUT_LAYER_NAME,
            type=model_variant_topology.INPUT_LAYER_NAME,
            is_input=True,
            is_output=False,
        )
        topology.add_layer(
            model_variant_topology.INPUT_LAYER_NAME,
            layer_descrip=input_layer_description,
        )

        output_layer_description = LayerDescription(
            name=model_variant_topology.OUTPUT_LAYER_NAME,
            type=model_variant_topology.OUTPUT_LAYER_NAME,
            is_input=False,
            is_output=True,
        )
        topology.add_layer(
            model_variant_topology.OUTPUT_LAYER_NAME,
            layer_descrip=output_layer_description,
        )

    @staticmethod
    def _add_topology_nodes(
        model_proto: onnx.ModelProto,
        topology: ModelVariantTopology,
    ) -> None:

        for node in model_proto.graph.node:
            if node.name == "":
                raise ValueError("Node name cannot be empty")

            layer_description = LayerDescription(
                name=node.name,
                type=node.op_type,
                is_input=False,
                is_output=False,
            )

            topology.add_layer(node.name, layer_descrip=layer_description)

    @staticmethod
    def _add_topology_edges(
        model_proto: onnx.ModelProto,
        topology: ModelVariantTopology,
    ) -> None:
        initializers_dict: dict[str, onnx.TensorProto] = {
            tensor.name: tensor for tensor in model_proto.graph.initializer
        }
        node_proto_dict: dict[str, onnx.NodeProto] = {
            node.name: node for node in model_proto.graph.node
        }

        per_layer_inputs: dict[str, set[str]] = {}
        per_layer_outputs: dict[str, set[str]] = {}
        for layer_key, layer_description in topology.layers().items():
            per_layer_inputs[layer_key] = OnnxModelTopologyBuilder._find_layer_inputs(
                layer_key,
                model_proto,
                node_proto_dict,
                initializers_dict,
                layer_description.is_input,
                layer_description.is_output,
            )
            per_layer_outputs[layer_key] = OnnxModelTopologyBuilder._find_layer_outputs(
                layer_key,
                model_proto,
                node_proto_dict,
                initializers_dict,
                layer_description.is_input,
                layer_description.is_output,
            )

        first_layer_name: str
        second_layer_name: str
        for first_layer_name in topology.layers().keys():
            first_layer_output_names: set[str] = per_layer_outputs[first_layer_name]
            for second_layer_name in topology.layers().keys():
                if first_layer_name == second_layer_name:
                    continue

                second_layer_input_names: set[str] = per_layer_inputs[second_layer_name]

                comm_elements = OnnxModelTopologyBuilder.__get_common_elements(
                    first_layer_output_names, second_layer_input_names
                )
                if len(comm_elements) > 0:
                    edge_info = EdgeDescription(
                        source=first_layer_name,
                        target=second_layer_name,
                        tensors=comm_elements,
                    )
                    edge_key = (first_layer_name, second_layer_name)
                    topology.add_edge(edge_key=edge_key, edge_descript=edge_info)

    @staticmethod
    def _find_layer_inputs(
        layer_name: str,
        model_proto: onnx.ModelProto,
        node_proto_dict: dict[str, onnx.NodeProto],
        initializers_dict: dict[str, onnx.TensorProto],
        is_input: bool,
        is_output: bool,
    ) -> set[str]:
        input_names: set[str] = set()

        if is_input:
            pass
        elif is_output:
            ## Output nodes are like Identity layers of the output
            for output in model_proto.graph.output:
                if not output.name:
                    # Optional output -> Omitted
                    continue
                else:
                    # This is an output of the node
                    input_names.add(output.name)
        else:
            node = node_proto_dict[layer_name]
            for input in node.input:
                if not input:
                    # Optional input -> Omitted
                    continue
                elif input not in initializers_dict:
                    # This input is an output of another node or model input
                    input_names.add(input)

        return input_names

    @staticmethod
    def _find_layer_outputs(
        layer_name: str,
        model_proto: onnx.ModelProto,
        node_proto_dict: dict[str, onnx.NodeProto],
        initializers_dict: dict[str, onnx.TensorProto],
        is_input: bool,
        is_output: bool,
    ) -> set[str]:
        output_names: set[str] = set()

        if is_input:
            ## Input nodes are like Identity layers of the model output
            for input in model_proto.graph.input:
                if not input.name:
                    # Optional input -> Omitted
                    continue

                if input.name not in initializers_dict:
                    # This input is not a weight
                    output_names.add(input.name)
        elif is_output:
            pass
        else:
            node = node_proto_dict[layer_name]
            for output in node.output:
                if not output:
                    # Optional output -> Omitted
                    continue
                else:
                    # This is an output of the node
                    output_names.add(output)

        return output_names

    @staticmethod
    def __get_common_elements(
        first_node_outs: set[str], second_node_ins: set[str]
    ) -> set[str]:
        common_elements = set()
        for elem in first_node_outs:
            if elem in second_node_ins:
                common_elements.add(elem)
        return common_elements
