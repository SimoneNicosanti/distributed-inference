import onnx
from distributed_inference.model_management.model_profile import profile_model
from distributed_inference.domain.ModelGraphInfo import ModelGraph, LayerInfo
from typing import cast
import onnxruntime as ort

from pathlib import Path
from tempfile import NamedTemporaryFile

import copy


def profile_model_with_aggregation(onnx_model: onnx.ModelProto) -> ModelGraph:

    basic_model_graph = profile_optimized_model(
        onnx_model, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
    extended_model_graph = profile_optimized_model(
        onnx_model, ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)

    agg_model_graph: ModelGraph = copy.deepcopy(extended_model_graph)

    compute_aggregate_model_info(
        agg_model_graph, basic_model_graph, extended_model_graph)

    return agg_model_graph


def compute_aggregate_model_info(agg_model_graph: ModelGraph, basic_model_graph: ModelGraph, extended_model_graph: ModelGraph) -> None:
    ext_to_process: dict[str, LayerInfo] = {}
    for ext_layer_name, ext_layer_info in extended_model_graph.get_all_layers().items():
        # There is a layer with the same name in the basic optimized
        if basic_model_graph.has_layer(ext_layer_name):
            basic_layer_info = basic_model_graph.get_layer_info(
                ext_layer_name)
            ext_layer_inputs, ext_layer_outputs = set(
                ext_layer_info.inputs.keys()), set(ext_layer_info.outputs.keys())
            basic_layer_inputs, basic_layer_outputs = set(
                basic_layer_info.inputs.keys()), set(basic_layer_info.outputs.keys())

            if ext_layer_inputs != basic_layer_inputs or ext_layer_outputs != basic_layer_outputs:
                # The I/O of the layer is not the same -> An aggregation has been done
                ext_to_process[ext_layer_name] = ext_layer_info
            else:
                # The I/O of the layer is the same -> No aggregation has been done
                # We restore the flops of the layer
                agg_model_graph.get_layer_info(
                    ext_layer_name).flops = basic_layer_info.flops

        # There is no layer with the same name in the basic optimized
        # An aggregation has been done
        else:
            ext_to_process[ext_layer_name] = ext_layer_info

    for ext_layer_name, ext_layer_info in ext_to_process.items():
        fused_layers = infer_fused_group(
            basic_model_graph, ext_layer_info)

        if fused_layers is None:
            continue

        fused_layers_info = basic_model_graph.gat_layer_info_from_iterable(
            fused_layers)

        agg_layer_info = agg_model_graph.get_layer_info(ext_layer_name)
        agg_layer_info.aggregated_layers = list(fused_layers_info.values())
        agg_layer_info.flops = sum(
            layer_info.flops for layer_info in fused_layers_info.values())
        # This is an aggregated node, but not in the sense of a coarsening
        agg_layer_info.is_aggregated = False


def infer_fused_group(
    basic_model_graph: ModelGraph,
    opt_layer_info: LayerInfo,
) -> set[str] | None:

    output_producers = {
        basic_model_graph.find_tensor_producer(tensor_name)
        for tensor_name in opt_layer_info.outputs.keys()
    }

    # If an output does not exist in the basic graph this is not a simple fusion
    # It is not easy to infer the fused group
    # We consider that this is not a fusion
    # WARNING: THIS SHOULD NOT HAPPEN WITH STANDARD
    if not output_producers or None in output_producers:
        return None

    producers = cast(set[str], output_producers)
    pending: list[str] = list(producers)
    fused_layers: set[str] = set()

    while pending:
        layer_name = pending.pop()

        if layer_name in fused_layers:
            continue

        fused_layers.add(layer_name)

        basic_layer_info = basic_model_graph.get_layer_info(layer_name)

        for tensor_name in basic_layer_info.inputs.keys():
            # Tensor is a boundary of the optimized node
            # We do need to process its generator
            if tensor_name in opt_layer_info.inputs.keys():
                continue

            producer = basic_model_graph.find_tensor_producer(tensor_name)

            if producer is None:
                # Abbiamo raggiunto un graph input che il nodo
                # ottimizzato non dichiara come input.
                return None

            pending.append(producer)

    return fused_layers


def profile_optimized_model(onnx_model: onnx.ModelProto, opt_level: ort.GraphOptimizationLevel) -> ModelGraph:

    with NamedTemporaryFile(suffix=".onnx") as tmp_file:
        base_model_path = Path(tmp_file.name)

        optimize_model(
            onnx_model,
            base_model_path,
            opt_level,
        )

        opt_onnx_model = onnx.load(base_model_path)
        profile_flops = False
        if opt_level == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC or opt_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL:
            profile_flops = True
        opt_model_graph = profile_model(
            opt_onnx_model, profile_flops=profile_flops)

        return opt_model_graph


def optimize_model(onnx_model: onnx.ModelProto, optimized_model_path: Path, opt_level: ort.GraphOptimizationLevel) -> None:

    sess_options = ort.SessionOptions()
    sess_options.optimized_model_filepath = optimized_model_path.as_posix()
    sess_options.graph_optimization_level = opt_level

    sess = ort.InferenceSession(onnx_model.SerializeToString(), sess_options)
    del sess
