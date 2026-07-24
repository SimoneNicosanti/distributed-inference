import copy
from pathlib import Path
from typing import cast, override

import numpy as np
import onnx
import onnx_tool
import sympy

from distributed_inference.application.model_profile.contracts.model_graph_extractor import (
    ModelGraphExtractor,
)
from distributed_inference.domain import model_graph_info
from distributed_inference.domain.model_graph_info import (
    DynamicShapeType,
    EdgeInfo,
    FlopsInfo,
    LayerInfo,
    ModelGraph,
    ModelInfo,
    TensorInfo,
)


class OnnxGraphExtractor(ModelGraphExtractor):
    def __init__(
        self,
    ):
        pass

    @override
    def extract_model_graph(
        self,
        path: Path,
        model_info: ModelInfo,
        profile_flops: bool,
        profile_tensors: bool,
    ) -> ModelGraph:

        model_proto = onnx.load(path.as_posix())

        inferred_model = OnnxGraphExtractor.__infer_model_shape(model_proto)
        model_graph: ModelGraph = ModelGraph(model_info=model_info)

        try:
            onnx.checker.check_model(inferred_model, full_check=True)
            # model_proto = infer_shapes(inferred_model)

            OnnxGraphExtractor.__init_model_graph(
                inferred_model, model_graph, model_info
            )
            OnnxGraphExtractor.__add_model_nodes(
                inferred_model, model_graph, model_info
            )
            if profile_flops:
                OnnxGraphExtractor.__add_layers_flops(
                    inferred_model, model_graph, model_info
                )
            OnnxGraphExtractor.__add_model_edges(
                inferred_model, model_graph, model_info
            )
            if profile_tensors:
                OnnxGraphExtractor.__add_tensors_info(
                    inferred_model, model_graph, model_info
                )
            OnnxGraphExtractor.__add_model_info(inferred_model, model_graph, model_info)
            OnnxGraphExtractor.__clear_model_graph(model_graph)

        except onnx.checker.ValidationError as e:
            raise Exception("Invalid ONNX model: " + str(e))

        return model_graph

    ## This is here because the way the fusions are done depends on the specific tool
    @override
    def aggregate_model_graphs(
        self, level_1_graph: ModelGraph, level_2_graph: ModelGraph
    ) -> ModelGraph:
        agg_model_graph: ModelGraph = copy.deepcopy(level_2_graph)

        ext_to_process: dict[str, LayerInfo] = {}
        for (
            ext_layer_name,
            ext_layer_info,
        ) in level_2_graph.get_all_layers().items():
            # There is a layer with the same name in the basic optimized
            if level_1_graph.has_layer(ext_layer_name):
                basic_layer_info = level_1_graph.get_layer_info(ext_layer_name)
                ext_layer_inputs, ext_layer_outputs = (
                    ext_layer_info.inputs,
                    ext_layer_info.outputs,
                )
                basic_layer_inputs, basic_layer_outputs = (
                    basic_layer_info.inputs,
                    basic_layer_info.outputs,
                )

                if (
                    ext_layer_inputs != basic_layer_inputs
                    or ext_layer_outputs != basic_layer_outputs
                ):
                    # The I/O of the layer is not the same -> An aggregation has been done
                    ext_to_process[ext_layer_name] = ext_layer_info
                else:
                    # The I/O of the layer is the same -> No aggregation has been done
                    # We restore the flops of the layer
                    agg_model_graph.get_layer_info(
                        ext_layer_name
                    ).flops = basic_layer_info.flops

            # There is no layer with the same name in the basic optimized
            # An aggregation has been done
            else:
                ext_to_process[ext_layer_name] = ext_layer_info

        for ext_layer_name, ext_layer_info in ext_to_process.items():
            fused_layers = OnnxGraphExtractor._infer_fused_group(
                level_1_graph, ext_layer_info
            )

            if fused_layers is None:
                continue

            fused_layers_info = level_1_graph.get_layer_info_from_iterable(fused_layers)

            agg_layer_info = agg_model_graph.get_layer_info(ext_layer_name)
            agg_layer_info.aggregated_layers = list(fused_layers_info.values())
            agg_layer_info.flops = sum(
                (layer_info.flops for layer_info in fused_layers_info.values()),
                start=FlopsInfo(),
            )
            # This is an aggregated node, but not in the sense of a coarsening
            agg_layer_info.is_aggregated = False

        agg_model_graph.set_tensors_map(level_1_graph.get_tensors_map())

        return agg_model_graph

    @staticmethod
    def _infer_fused_group(
        basic_model_graph: ModelGraph,
        opt_layer_info: LayerInfo,
    ) -> set[str] | None:

        output_producers = {
            basic_model_graph.find_tensor_producer(tensor_name)
            for tensor_name in opt_layer_info.outputs
        }

        # If an output does not exist in the basic graph this is not a simple fusion
        # It is not easy to infer the fused group
        # We consider that this is not a fusion
        # NOTE: THIS SHOULD NOT HAPPEN WITH STANDARD OPTIMIZATIONS
        if not output_producers or None in output_producers:
            return None

        producers = cast(set[str], output_producers)
        pending: list[str] = list(producers)
        fused_layers: set[str] = set()

        while pending:
            layer_name = pending.pop()

            if layer_name in fused_layers:
                continue

            if basic_model_graph.get_layer_info(layer_name).is_input:
                continue

            fused_layers.add(layer_name)

            basic_layer_info = basic_model_graph.get_layer_info(layer_name)

            for tensor_name in basic_layer_info.inputs:
                # Tensor is a boundary of the optimized node
                # We do need to process its generator
                if tensor_name in opt_layer_info.inputs:
                    continue

                producer = basic_model_graph.find_tensor_producer(tensor_name)

                if producer is None:
                    # Abbiamo raggiunto un graph input che il nodo
                    # ottimizzato non dichiara come input.
                    return None

                pending.append(producer)

        return fused_layers

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
    def __clear_model_graph(model_graph: ModelGraph) -> None:
        # Removing all nodes that are not reachable from the input node
        # In some cases, there are nodes used to define the weights
        # Or pre-processing operations on them (likq quantization)
        reachable = model_graph.get_reachable_from_layer(
            model_graph_info.INPUT_LAYER_NAME
        ).keys()

        all_layers = model_graph.get_all_layers().keys()

        unreachable = set(all_layers) - set(reachable)

        model_graph.remove_layers_from_iterable(unreachable)

    @staticmethod
    def __init_model_graph(
        model_proto: onnx.ModelProto, model_graph: ModelGraph, model_info: ModelInfo
    ) -> None:

        input_names: set[str] = set()
        for input in model_proto.graph.input:
            input_names.add(input.name)

        input_layer_info = LayerInfo(
            name=model_graph_info.INPUT_LAYER_NAME,
            type=model_graph_info.INPUT_LAYER_NAME,
            flops=FlopsInfo(
                flops={seq_size: 0 for seq_size in model_info.sequence_sizes}
            ),
            weights_size=0,
            inputs=input_names,
            outputs=input_names,
            is_input=True,
            is_output=False,
            is_aggregated=False,
            aggregated_layers=[],
        )
        model_graph.add_layer(layer_info=input_layer_info)

        output_names: set[str] = set()
        for output in model_proto.graph.output:
            output_names.add(output.name)

        output_layer_info = LayerInfo(
            name=model_graph_info.OUTPUT_LAYER_NAME,
            type=model_graph_info.OUTPUT_LAYER_NAME,
            flops=FlopsInfo(
                flops={seq_size: 0 for seq_size in model_info.sequence_sizes}
            ),
            weights_size=0,
            inputs=output_names,
            outputs=output_names,
            is_input=False,
            is_output=True,
            is_aggregated=False,
            aggregated_layers=[],
        )
        model_graph.add_layer(layer_info=output_layer_info)

    @staticmethod
    def __add_model_nodes(
        model_proto: onnx.ModelProto,
        model_graph: ModelGraph,
        model_info: ModelInfo,
    ) -> None:

        ## Dict of model weights
        ## We need it to distinguish between node inputs and weights
        initializers_dict: dict[str, onnx.TensorProto] = {
            tensor.name: tensor for tensor in model_proto.graph.initializer
        }

        for node in model_proto.graph.node:
            input_names, output_names, weights_size = (
                OnnxGraphExtractor.__extract_node_info(
                    initializers_dict,
                    node,
                    model_info,
                )
            )

            layer_info = LayerInfo(
                name=node.name,
                type=node.op_type,
                flops=FlopsInfo(flops={}),  # type: ignore
                weights_size=weights_size,
                inputs=input_names,
                outputs=output_names,
                is_input=False,
                is_output=False,
                is_aggregated=False,
                aggregated_layers=[],
            )

            model_graph.add_layer(layer_info=layer_info)

        pass

    @staticmethod
    def __add_tensors_info(
        model_proto: onnx.ModelProto,
        model_graph: ModelGraph,
        model_info: ModelInfo,
    ) -> None:

        all_tensors: set[str] = set()
        for _, layer_info in model_graph.get_all_layers().items():
            all_tensors.update(layer_info.inputs)
            all_tensors.update(layer_info.outputs)

        per_name_tensor_info_dict: dict[str, TensorInfo] = {}
        for tensor in all_tensors:
            tensor_info = TensorInfo(
                name=tensor,
                shapes={seq_size: [] for seq_size in model_info.sequence_sizes},
                sizes={seq_size: 0 for seq_size in model_info.sequence_sizes},
            )
            per_name_tensor_info_dict.setdefault(tensor, tensor_info)

        ## Computing tensor dims with different sizes
        for seq_size in model_info.sequence_sizes:
            tool_model = onnx_tool.Model(
                model_proto,
            )
            tool_inputs = {}
            for input in model_proto.graph.input:
                shape, dtype = OnnxGraphExtractor.__compute_value_info_proto_info(
                    input, model_info, seq_size
                )
                tool_inputs[input.name] = np.zeros(shape, dtype=dtype)

            tool_graph = tool_model.graph
            tool_graph.shape_infer(tool_inputs)

            for tensor in all_tensors:
                tool_tensor: onnx_tool.tensor.Tensor = tool_model.graph.tensormap[
                    tensor
                ]

                shape, size = tool_tensor.get_shape(), tool_tensor.get_memsize()
                per_name_tensor_info_dict[tensor].shapes[seq_size] = shape
                per_name_tensor_info_dict[tensor].sizes[seq_size] = size

        model_graph.set_tensors_map(per_name_tensor_info_dict)
        pass

    @staticmethod
    def __add_layers_flops(
        model_proto: onnx.ModelProto, model_graph: ModelGraph, model_info: ModelInfo
    ) -> None:

        per_layer_flops_info: dict[str, FlopsInfo] = {}
        for layer in model_graph.get_all_layers().keys():
            flops_info = FlopsInfo(
                flops={seq_size: 0 for seq_size in model_info.sequence_sizes}
            )
            per_layer_flops_info.setdefault(layer, flops_info)

        ## Computing flops with different sizes
        for seq_size in model_info.sequence_sizes:
            tool_model = onnx_tool.Model(model_proto)
            tool_inputs = {}
            for input in model_proto.graph.input:
                shape, dtype = OnnxGraphExtractor.__compute_value_info_proto_info(
                    input, model_info, seq_size
                )
                tool_inputs[input.name] = np.zeros(shape, dtype=dtype)

            tool_graph = tool_model.graph
            tool_graph.shape_infer(tool_inputs)
            tool_graph.profile()

            for node_name, node in tool_graph.nodemap.items():
                # Maybe we can get the aggregated static memory, but let's do it manually for now
                # "memory_bytes": int(node.memory),
                per_layer_flops_info[node_name].flops[seq_size] = int(2 * node.macs[0])

        for layer, flops_info in per_layer_flops_info.items():
            model_graph.get_layer_info(layer).flops = flops_info

        pass

    @staticmethod
    def __extract_node_info(
        initializers_dict: dict[str, onnx.TensorProto],
        node: onnx.NodeProto,
        model_info: ModelInfo,
    ) -> tuple[set[str], set[str], float]:

        input_names: set[str] = set()
        output_names: set[str] = set()
        weights_size: float = 0

        for input_name in node.input:
            if not input_name:
                # Optional input -> Omitted
                continue

            if input_name in initializers_dict.keys():
                # This input is a weight
                weights_size += OnnxGraphExtractor.__compute_tensor_size(
                    initializers_dict[input_name], model_info
                )
            else:
                # This input is an output of another node or model input
                input_names.add(input_name)

        for output_name in node.output:
            if not output_name:
                # Optional output -> Omitted
                continue
            else:
                # This is an output of the node
                output_names.add(output_name)

        return input_names, output_names, weights_size

    @staticmethod
    def __add_model_edges(
        model_proto: onnx.ModelProto, model_graph: ModelGraph, model_info: ModelInfo
    ) -> None:
        first_layer_name: str
        second_layer_name: str
        for first_layer_name, first_layer_info in model_graph.get_all_layers().items():
            first_out_names: set[str] = first_layer_info.outputs

            for (
                second_layer_name,
                second_layer_info,
            ) in model_graph.get_all_layers().items():
                if first_layer_name == second_layer_name:
                    continue

                second_in_names: set[str] = second_layer_info.inputs

                comm_elements = OnnxGraphExtractor.__get_common_elements(
                    first_out_names, second_in_names
                )
                if len(comm_elements) > 0:
                    tensors = comm_elements

                    edge_info = EdgeInfo(
                        source=first_layer_name,
                        target=second_layer_name,
                        tensors=tensors,
                    )
                    model_graph.add_edge(edge_info=edge_info)
        pass

    @staticmethod
    def __get_common_elements(
        first_node_outs: set[str], second_node_ins: set[str]
    ) -> set[str]:
        common_elements = set()
        for elem in first_node_outs:
            if elem in second_node_ins:
                common_elements.add(elem)
        return common_elements

    @staticmethod
    def __add_model_info(
        model_proto: onnx.ModelProto, model_graph: ModelGraph, model_info: ModelInfo
    ) -> None:
        pass

    @staticmethod
    def __compute_tensor_size(
        tensor: onnx.ValueInfoProto | onnx.TensorProto,
        model_info: ModelInfo,
        sequence_size: int = 1,
    ) -> float:
        ## NOTE: Size is computed in bytes since we do not know yet how it will be used
        if isinstance(tensor, onnx.ValueInfoProto):
            shape, dtype = OnnxGraphExtractor.__compute_value_info_proto_info(
                tensor, model_info, sequence_size
            )
            total_entries = int(np.prod(shape))
            return total_entries * dtype.itemsize

        elif isinstance(tensor, onnx.TensorProto):
            array = onnx.numpy_helper.to_array(tensor)
            return int(array.nbytes)

    @staticmethod
    def __compute_value_info_proto_info(
        value_info_proto: onnx.ValueInfoProto,
        model_info: ModelInfo,
        sequence_size: int = 1,
    ) -> tuple[list[int], np.dtype]:
        # bindings = {
        #     "batch_size": model_info.batch_size,
        #     "sequence_size": sequence_size,
        #     "sequence_length": sequence_size,
        # }

        bindings: dict[str, int] = {}
        for dyn_shape_name, dyn_shape_type in model_info.dynamic_shapes.items():
            match dyn_shape_type:
                case DynamicShapeType.BATCH:
                    bindings[dyn_shape_name] = 1
                case DynamicShapeType.SEQUENCE:
                    bindings[dyn_shape_name] = sequence_size
                case _:
                    raise ValueError(f"Unknown dynamic shape type: {dyn_shape_type}")

        shape = [
            OnnxGraphExtractor._resolve_dimension(dim, bindings)
            for dim in value_info_proto.type.tensor_type.shape.dim
        ]

        dtype = np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(
                value_info_proto.type.tensor_type.elem_type
            )
        )

        return shape, dtype

    @staticmethod
    def _resolve_dimension(
        dim: onnx.TensorShapeProto.Dimension,
        bindings: dict[str, int],
    ) -> int:
        dimension_type = dim.WhichOneof("value")

        if dimension_type == "dim_value":
            return dim.dim_value

        if dimension_type != "dim_param":
            raise ValueError("Anonymous dynamic dimension cannot be resolved")

        expression = sympy.sympify(dim.dim_param)

        substitutions = {sympy.Symbol(name): value for name, value in bindings.items()}

        resolved = expression.subs(substitutions)

        if resolved.free_symbols:
            raise ValueError(f"Unresolved symbolic dimension: {dim.dim_param}")

        if not resolved.is_integer:
            raise ValueError(f"Non-integer symbolic dimension: {dim.dim_param}")

        return int(resolved)
