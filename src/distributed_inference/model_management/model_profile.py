import onnx

from distributed_inference.domain.ModelGraphInfo import LayerInfo, EdgeInfo, ModelGraph
from distributed_inference.domain import ModelGraphInfo
import numpy as np
import onnx_tool


def profile_model(model_proto: onnx.ModelProto, profile_flops: bool = True) -> ModelGraph:

    # TODO: Add support for model information
    model_graph: ModelGraph = ModelGraph()

    try:
        onnx.checker.check_model(model_proto)

        __init_model_graph(model_proto, model_graph)
        __add_model_nodes(model_proto, model_graph, profile_flops)
        __add_model_edges(model_proto, model_graph)
        __add_model_info(model_proto, model_graph)
        __clear_model_graph(model_graph)

    except onnx.checker.ValidationError as e:
        raise Exception("Invalid ONNX model: " + str(e))

    return model_graph


def __clear_model_graph(model_graph: ModelGraph) -> None:
    # Removing all nodes that are not reachable from the input node
    # In some cases, there are nodes used to define the weights
    # Or pre-processing operations on them (likq quantization)
    reachable = model_graph.get_reachable_from_layer(
        ModelGraphInfo.INPUT_LAYER_NAME).keys()

    all_layers = model_graph.get_all_layers().keys()

    unreachable = set(all_layers) - set(reachable)

    model_graph.remove_layers_from_iterable(unreachable)


def __init_model_graph(model_proto: onnx.ModelProto, model_graph: ModelGraph) -> None:

    input_names: list[str] = []
    input_sizes: list[float] = []
    for input in model_proto.graph.input:
        input_names.append(input.name)
        input_size = __compute_tensor_size(input)
        input_sizes.append(input_size)

    input_layer_info = LayerInfo(name=ModelGraphInfo.INPUT_LAYER_NAME,
                                 type=ModelGraphInfo.INPUT_LAYER_NAME,
                                 flops=0,
                                 weights_size=0,
                                 inputs={
                                     input_name: input_size for input_name, input_size in zip(input_names, input_sizes)},
                                 outputs={
                                     input_name: input_size for input_name, input_size in zip(input_names, input_sizes)},
                                 is_input=True,
                                 is_output=False,
                                 is_aggregated=False,
                                 aggregated_layers=[])
    model_graph.add_layer(layer_info=input_layer_info)

    output_names: list[str] = []
    output_sizes: list[float] = []
    for output in model_proto.graph.output:
        output_names.append(output.name)
        output_size = __compute_tensor_size(output)
        output_sizes.append(output_size)

    output_layer_info = LayerInfo(name=ModelGraphInfo.OUTPUT_LAYER_NAME,
                                  type=ModelGraphInfo.OUTPUT_LAYER_NAME,
                                  flops=0,
                                  weights_size=0,
                                  inputs={
                                      output_name: output_size
                                      for output_name, output_size in zip(output_names, output_sizes)},
                                  outputs={
                                      output_name: output_size
                                      for output_name, output_size in zip(output_names, output_sizes)
                                  },
                                  is_input=False,
                                  is_output=True,
                                  is_aggregated=False,
                                  aggregated_layers=[])
    model_graph.add_layer(layer_info=output_layer_info)


def __add_model_nodes(model_proto: onnx.ModelProto, model_graph: ModelGraph, profile_flops: bool) -> None:

    flops_dict: dict[str, float] = {}
    if profile_flops:
        flops_dict = __build_flops_dict(model_proto)

    model_inputs: dict[str, onnx.ValueInfoProto] = {
        input.name: input
        for input in model_proto.graph.input
    }
    model_outputs: dict[str, onnx.ValueInfoProto] = {
        output.name: output
        for output in model_proto.graph.output
    }
    initializers_dict: dict[str, onnx.TensorProto] = {
        tensor.name: tensor for tensor in model_proto.graph.initializer}
    value_info_proto_dict: dict[str, onnx.ValueInfoProto] = {
        tensor.name: tensor for tensor in model_proto.graph.value_info}

    for node in model_proto.graph.node:

        input_names, input_sizes, output_names, output_sizes, weights_size = __extract_node_info(
            initializers_dict, value_info_proto_dict, model_inputs, model_outputs, node)

        layer_info = LayerInfo(name=node.name,
                               type=node.op_type,
                               flops=flops_dict.get(node.name, 0),
                               weights_size=weights_size,
                               inputs={input_name: input_size for input_name,
                                       input_size in zip(input_names, input_sizes)},
                               outputs={output_name: output_size for output_name,
                                        output_size in zip(output_names, output_sizes)},
                               is_input=False,
                               is_output=False,
                               is_aggregated=False,
                               aggregated_layers=[])

        model_graph.add_layer(layer_info=layer_info)
    pass


def __extract_node_info(
    initializers_dict: dict[str, onnx.TensorProto],
    value_info_proto_dict: dict[str, onnx.ValueInfoProto],
    model_inputs: dict[str, onnx.ValueInfoProto],
    model_outputs: dict[str, onnx.ValueInfoProto],
    node: onnx.NodeProto
) -> tuple[list[str], list[float], list[str], list[float], float]:

    input_names: list[str] = []
    input_sizes: list[float] = []
    output_names: list[str] = []
    output_sizes: list[float] = []
    weights_size: float = 0

    for input_name in node.input:
        if input_name in initializers_dict.keys():
            # This input is a weight
            weights_size += __compute_tensor_size(
                initializers_dict[input_name])

        elif input_name in value_info_proto_dict.keys():
            # This input is an activation
            input_names.append(input_name)
            input_size = __compute_tensor_size(
                value_info_proto_dict[input_name])
            input_sizes.append(input_size)

        elif input_name in model_inputs.keys():
            # This input is an input of the whole model
            input_names.append(input_name)
            input_size = __compute_tensor_size(
                model_inputs[input_name])
            input_sizes.append(input_size)

    for output_name in node.output:
        if output_name in value_info_proto_dict.keys():
            # This output is an activation (obviously)
            output_names.append(output_name)
            output_size = __compute_tensor_size(
                value_info_proto_dict[output_name])
            output_sizes.append(output_size)

        elif output_name in model_outputs.keys():
            # This output is an output of the whole model
            output_names.append(output_name)
            output_size = __compute_tensor_size(
                model_outputs[output_name])
            output_sizes.append(output_size)

    return input_names, input_sizes, output_names, output_sizes, weights_size


def __add_model_edges(model_proto: onnx.ModelProto, model_graph: ModelGraph) -> None:
    first_layer_name: str
    second_layer_name: str
    for first_layer_name, first_layer_info in model_graph.get_all_layers().items():
        first_out_names: set[str] = set(
            list(first_layer_info.outputs.keys()))

        for second_layer_name, second_layer_info in model_graph.get_all_layers().items():
            if first_layer_name == second_layer_name:
                continue

            second_in_names: set[str] = set(
                list(second_layer_info.inputs.keys()))

            comm_elements = __get_common_elements(
                first_out_names, second_in_names
            )
            if len(comm_elements) > 0:
                tensors = {
                    act_name: act_size for act_name, act_size in second_layer_info.inputs.items() if act_name in comm_elements}

                edge_info = EdgeInfo(source=first_layer_name,
                                     target=second_layer_name,
                                     tensors=tensors)
                model_graph.add_edge(edge_info=edge_info)
    pass


def __get_common_elements(first_node_outs: set[str], second_node_ins: set[str]) -> set[str]:
    common_elements = set()
    for elem in first_node_outs:
        if elem in second_node_ins:
            common_elements.add(elem)
    return common_elements


def __add_model_info(model_proto: onnx.ModelProto, model_graph: ModelGraph) -> None:
    pass


def __compute_tensor_size(tensor: onnx.ValueInfoProto | onnx.TensorProto) -> float:

    if isinstance(tensor, onnx.ValueInfoProto):
        shape, dtype = __compute_value_info_proto_info(tensor)
        total_entries = int(np.prod(shape))
        return total_entries * dtype.itemsize

    elif isinstance(tensor, onnx.TensorProto):
        array = onnx.numpy_helper.to_array(tensor)
        return int(array.nbytes)


def __compute_value_info_proto_info(value_info_proto: onnx.ValueInfoProto) -> tuple[list[int], np.dtype]:
    shape: list[int] = []
    for dim in value_info_proto.type.tensor_type.shape.dim:
        curr_dim = 1
        if dim.dim_value != -1 and dim.dim_value != 0:
            curr_dim = dim.dim_value
        shape.append(curr_dim)

    dtype = np.dtype(
        onnx.helper.tensor_dtype_to_np_dtype(
            value_info_proto.type.tensor_type.elem_type)
    )

    return shape, dtype


def __build_flops_dict(model_proto: onnx.ModelProto) -> dict[str, float]:
    flops_dict: dict[str, float] = {}

    tool_model = onnx_tool.Model(model_proto)

    tool_inputs = {}
    for input in model_proto.graph.input:
        shape, dtype = __compute_value_info_proto_info(input)
        tool_inputs[input.name] = np.zeros(shape, dtype=dtype)

    tool_graph = tool_model.graph
    tool_graph.shape_infer(tool_inputs)
    tool_graph.profile()

    for node_name, node in tool_graph.nodemap.items():
        # Maybe we can get the aggregated static memory, but let's do it manually for now
        # "memory_bytes": int(node.memory),
        flops_dict[node_name] = int(2 * node.macs[0])

    return flops_dict
