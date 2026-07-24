from distributed_inference.application.model_profile.profiling.model_profile import (
    profile_model,
)
from distributed_inference.domain.model_graph_info import (
    ModelGraph,
    LayerInfo,
    ModelInfo,
    FlopsInfo,
)
from typing import cast


from distributed_inference.application.model_profile.optimization.model_optimize import (
    optimize_model,
    OptimizationLevel,
)

import copy

from pathlib import Path
from tempfile import TemporaryDirectory

import onnx


def compute_aggregate_model_graph(
    basic_model_graph: ModelGraph,
    extended_model_graph: ModelGraph,
) -> ModelGraph:

    agg_model_graph: ModelGraph = copy.deepcopy(extended_model_graph)

    ext_to_process: dict[str, LayerInfo] = {}
    for ext_layer_name, ext_layer_info in extended_model_graph.get_all_layers().items():
        # There is a layer with the same name in the basic optimized
        if basic_model_graph.has_layer(ext_layer_name):
            basic_layer_info = basic_model_graph.get_layer_info(ext_layer_name)
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
        fused_layers = _infer_fused_group(basic_model_graph, ext_layer_info)

        if fused_layers is None:
            continue

        fused_layers_info = basic_model_graph.get_layer_info_from_iterable(fused_layers)

        agg_layer_info = agg_model_graph.get_layer_info(ext_layer_name)
        agg_layer_info.aggregated_layers = list(fused_layers_info.values())
        agg_layer_info.flops = sum(
            (layer_info.flops for layer_info in fused_layers_info.values()),
            start=FlopsInfo(),
        )
        # This is an aggregated node, but not in the sense of a coarsening
        agg_layer_info.is_aggregated = False

    agg_model_graph.set_tensors_map(basic_model_graph.get_tensors_map())

    return agg_model_graph


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


def profile_with_model_optimization(
    input_path: Path,
    model_info: ModelInfo,
    opt_level: OptimizationLevel,
) -> ModelGraph:

    with TemporaryDirectory() as tmp_dir:
        opt_path = Path(tmp_dir).joinpath(f"{input_path.stem}.onnx")

        optimize_model(
            input_path=input_path,
            output_path=opt_path,
            model_info=model_info,
            opt_level=opt_level,
        )

        opt_onnx_model = onnx.load(opt_path.as_posix())

        profile_flops = False
        profile_tensors = False
        if opt_level == OptimizationLevel.BASIC or opt_level == OptimizationLevel.NONE:
            profile_flops = True
            profile_tensors = True

        opt_model_graph = profile_model(
            opt_onnx_model,
            model_info,
            profile_flops=profile_flops,
            profile_tensors=profile_tensors,
        )

        return opt_model_graph
