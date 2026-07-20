from distributed_inference.model_management.model_profile import profile_model
from distributed_inference.model_management.model_profile_agg import (
    profile_model_with_aggregation,
)
import onnx
import networkx as nx
import onnxruntime as ort
from distributed_inference.domain.ModelGraphInfo import LayerInfo
import onnxsim
from onnx import version_converter


def ensure_opset(model: onnx.ModelProto, target_opset: int = 17) -> onnx.ModelProto:
    current_opset = next(
        opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")
    )

    if current_opset < target_opset:
        model = version_converter.convert_version(model, target_opset)

    onnx.checker.check_model(model)
    return model


def count_ideals(graph: nx.DiGraph) -> int:
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The graph must be a DAG.")

    return sum(1 for _ in nx.antichains(graph))


MODEL_NAME = "vit-base"

onnx_model = onnx.load(f"./../../../assets/models/{MODEL_NAME}.onnx")
onnx_model = ensure_opset(onnx_model)

agg_model_graph = profile_model_with_aggregation(onnx_model)

# base_model_graph = profile_model(onnx_model)

print("Numero di nodi Agg: " + str(len(agg_model_graph.get_all_layers())))
print("Numero di archi Agg: " + str(len(agg_model_graph.get_all_edges())))

for layer, layer_info in agg_model_graph.get_all_layers().items():
    print(layer_info)
    print()
