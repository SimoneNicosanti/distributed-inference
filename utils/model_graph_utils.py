import re
from pathlib import Path

import networkx as nx
import onnx
from onnx import TensorProto, helper

from distributed_inference.domain.model_graph_info import (
    LayerKey,
    ModelGraph,
)

_VISUALIZATION_DOMAIN = "distributed_inference.visualization"


def export_model_graph_to_dummy_onnx(
    model_graph: ModelGraph,
    output_path: str | Path,
) -> onnx.ModelProto:
    """
    Export a ModelGraph as a dummy ONNX model for visualization in Netron.

    The generated model is structurally valid but not executable, since its
    nodes belong to a custom visualization domain.
    """
    layers = model_graph.get_all_layers()
    edges = model_graph.get_all_edges()

    if not layers:
        raise ValueError("Cannot export an empty ModelGraph")

    topology = nx.DiGraph()
    topology.add_nodes_from(layers)
    topology.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(topology):
        raise ValueError("ModelGraph must be a DAG")

    node_inputs: dict[LayerKey, list[str]] = {layer: [] for layer in layers}
    node_outputs: dict[LayerKey, list[str]] = {layer: [] for layer in layers}

    used_value_names: set[str] = set()

    # A tensor produced by one layer can be consumed by several layers.
    tensor_values: dict[tuple[LayerKey, str], str] = {}

    graph_inputs: list[onnx.ValueInfoProto] = []
    graph_outputs: list[onnx.ValueInfoProto] = []
    intermediate_values: list[onnx.ValueInfoProto] = []

    for (source, target), edge_info in edges.items():
        if source not in layers:
            raise ValueError(f"Unknown edge source: {source!r}")

        if target not in layers:
            raise ValueError(f"Unknown edge target: {target!r}")

        edge_tensors = edge_info.tensors or {f"{source}_to_{target}"}

        for tensor_name in edge_tensors:
            tensor_key = (source, tensor_name)
            tensor_size = model_graph.get_default_sizes_for_tensors_set({tensor_name})[
                tensor_name
            ]

            onnx_value_name = tensor_values.get(tensor_key)

            if onnx_value_name is None:
                onnx_value_name = _make_unique_identifier(
                    f"{source}__{tensor_name}",
                    used_value_names,
                )

                node_outputs[source].append(onnx_value_name)

                intermediate_values.append(
                    _make_dummy_value_info(
                        value_name=onnx_value_name,
                        original_tensor_name=tensor_name,
                        tensor_size=tensor_size,
                    )
                )

            node_inputs[target].append(onnx_value_name)

    onnx_nodes: list[onnx.NodeProto] = []

    for layer_key in nx.topological_sort(topology):
        layer_info = layers[layer_key]

        inputs = list(dict.fromkeys(node_inputs[layer_key]))
        outputs = list(dict.fromkeys(node_outputs[layer_key]))

        if not inputs:
            declared_inputs = layer_info.inputs or {f"{layer_key}_input"}

            for tensor_name in declared_inputs:
                value_name = _make_unique_identifier(
                    f"graph_input__{layer_key}__{tensor_name}",
                    used_value_names,
                )

                inputs.append(value_name)

                graph_inputs.append(
                    _make_dummy_value_info(
                        value_name=value_name,
                        original_tensor_name=tensor_name,
                        tensor_size=tensor_size,
                    )
                )

        if not outputs:
            declared_outputs = layer_info.outputs or {f"{layer_key}_output"}

            for tensor_name in declared_outputs:
                value_name = _make_unique_identifier(
                    f"graph_output__{layer_key}__{tensor_name}",
                    used_value_names,
                )

                outputs.append(value_name)

                graph_outputs.append(
                    _make_dummy_value_info(
                        value_name=value_name,
                        original_tensor_name=tensor_name,
                        tensor_size=tensor_size,
                    )
                )

        attributes: dict[str, object] = {
            "original_type": layer_info.type,
            "flops": model_graph.get_default_flops_for_layer_set({layer_info.name})[
                layer_info.name
            ],
            "weights_size": float(layer_info.weights_size),
            "total_input_size": float(
                sum(
                    model_graph.get_default_sizes_for_tensors_set(
                        layer_info.inputs
                    ).values()
                )
            ),
            "total_output_size": float(
                sum(
                    model_graph.get_default_sizes_for_tensors_set(
                        layer_info.outputs
                    ).values()
                )
            ),
            "is_input": int(layer_info.is_input),
            "is_output": int(layer_info.is_output),
            "is_aggregated": int(layer_info.is_aggregated),
        }

        if layer_info.aggregated_layers:
            attributes["aggregated_layers"] = [
                layer.name for layer in layer_info.aggregated_layers
            ]

        node = helper.make_node(
            op_type=_safe_identifier(
                layer_info.type,
                fallback="Layer",
            ),
            inputs=inputs,
            outputs=outputs,
            name=layer_info.name,
            domain=_VISUALIZATION_DOMAIN,
            **attributes,  # type: ignore
        )

        onnx_nodes.append(node)

    model_name = (
        model_graph.model_info.name
        if model_graph.model_info is not None
        else "ModelGraphVisualization"
    )

    onnx_graph = helper.make_graph(
        nodes=onnx_nodes,
        name=model_name,
        inputs=graph_inputs,
        outputs=graph_outputs,
        value_info=intermediate_values,
    )

    onnx_model = helper.make_model(
        onnx_graph,
        producer_name="distributed-inference",
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid(_VISUALIZATION_DOMAIN, 1),
        ],
    )

    if model_graph.model_info is not None:
        helper.set_model_props(
            onnx_model,
            {
                "model_name": model_graph.model_info.name,
                "model_type": model_graph.model_info.type.value,
                "task": model_graph.model_info.task.value,
                "accuracy": str(model_graph.model_info.accuracy),
                "purpose": "Netron visualization only",
            },
        )

    onnx.checker.check_model(onnx_model)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    onnx.save_model(
        onnx_model,
        output_path.as_posix(),
    )

    return onnx_model


def _make_dummy_value_info(
    value_name: str,
    original_tensor_name: str,
    tensor_size: float,
) -> onnx.ValueInfoProto:
    # UINT8[size] makes the dimension correspond approximately
    # to the tensor size in bytes.
    dummy_length = max(1, round(tensor_size))

    value_info = helper.make_tensor_value_info(
        value_name,
        TensorProto.UINT8,
        [dummy_length],
    )

    value_info.doc_string = (
        f"Original tensor: {original_tensor_name}; size: {tensor_size:g} bytes"
    )

    return value_info


def _make_unique_identifier(
    value: str,
    used_identifiers: set[str],
) -> str:
    base = _safe_identifier(value, fallback="value")
    identifier = base
    suffix = 1

    while identifier in used_identifiers:
        identifier = f"{base}_{suffix}"
        suffix += 1

    used_identifiers.add(identifier)

    return identifier


def _safe_identifier(
    value: str,
    fallback: str,
) -> str:
    identifier = re.sub(
        r"[^A-Za-z0-9_]",
        "_",
        value,
    )

    if not identifier:
        identifier = fallback

    if identifier[0].isdigit():
        identifier = f"_{identifier}"

    return identifier
