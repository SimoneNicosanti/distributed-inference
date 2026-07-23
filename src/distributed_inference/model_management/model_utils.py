import onnx
import networkx as nx

from distributed_inference.domain.ModelGraphInfo import (
    ModelGraph,
    LayerKey,
    EdgeKey,
)
from pathlib import Path

from onnx.external_data_helper import uses_external_data


from onnx import helper, TensorProto
import re


def ensure_opset_to_path(
    input_path: str | Path,
    output_path: str | Path,
    target_opset: int,
) -> Path:
    """
    Ensure that the default ONNX opset is at least `target_opset` and save
    the resulting model to `output_path`.

    If the input uses external data, the output uses a single adjacent
    `<output_name>.data` file.
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {input_path}")

    if input_path == output_path:
        raise ValueError("input_path and output_path must be different")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Inspect the original representation without loading the weights.
    metadata_model = onnx.load(
        input_path,
        load_external_data=False,
    )

    current_opset = next(
        (
            opset.version
            for opset in metadata_model.opset_import
            if opset.domain in {"", "ai.onnx"}
        ),
        None,
    )

    if current_opset is None:
        raise ValueError("The model does not import the default ONNX opset")

    uses_external_weights = any(
        uses_external_data(initializer)
        for initializer in metadata_model.graph.initializer
    )

    # The converter may need the initializer values, so load external data.
    model = onnx.load(
        input_path,
        load_external_data=True,
    )

    if current_opset < target_opset:
        model = onnx.version_converter.convert_version(
            model,
            target_opset,
        )

    # Remove stale outputs from previous executions.
    output_path.unlink(missing_ok=True)

    external_data_path = output_path.with_name(f"{output_path.name}.data")
    external_data_path.unlink(missing_ok=True)

    if uses_external_weights:
        onnx.save_model(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
            size_threshold=0,
            # Keep Constant tensor attributes embedded: this avoids
            # shape-inference problems previously encountered with ORT.
            convert_attribute=False,
        )
    else:
        onnx.save_model(
            model,
            output_path,
            save_as_external_data=False,
        )

    # Checking through the path lets ONNX resolve adjacent external data.
    onnx.checker.check_model(
        output_path.as_posix(),
        full_check=True,
    )

    return output_path


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
    tensor_sizes: dict[tuple[LayerKey, str], float] = {}

    graph_inputs: list[onnx.ValueInfoProto] = []
    graph_outputs: list[onnx.ValueInfoProto] = []
    intermediate_values: list[onnx.ValueInfoProto] = []

    for (source, target), edge_info in edges.items():
        if source not in layers:
            raise ValueError(f"Unknown edge source: {source!r}")

        if target not in layers:
            raise ValueError(f"Unknown edge target: {target!r}")

        edge_tensors = edge_info.tensors or {f"{source}_to_{target}": 1.0}

        for tensor_name, tensor_size in edge_tensors.items():
            tensor_key = (source, tensor_name)

            previous_size = tensor_sizes.get(tensor_key)

            if previous_size is not None and previous_size != tensor_size:
                raise ValueError(
                    f"Inconsistent size for tensor {tensor_name!r} "
                    f"produced by {source!r}: "
                    f"{previous_size} != {tensor_size}"
                )

            onnx_value_name = tensor_values.get(tensor_key)

            if onnx_value_name is None:
                onnx_value_name = _make_unique_identifier(
                    f"{source}__{tensor_name}",
                    used_value_names,
                )

                tensor_values[tensor_key] = onnx_value_name
                tensor_sizes[tensor_key] = tensor_size

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
            declared_inputs = layer_info.inputs or {f"{layer_key}_input": 1.0}

            for tensor_name, tensor_size in declared_inputs.items():
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
            declared_outputs = layer_info.outputs or {f"{layer_key}_output": 1.0}

            for tensor_name, tensor_size in declared_outputs.items():
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
            "flops": float(layer_info.flops),
            "weights_size": float(layer_info.weights_size),
            "total_input_size": float(sum(layer_info.inputs.values())),
            "total_output_size": float(sum(layer_info.outputs.values())),
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
            **attributes,
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
