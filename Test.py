from src.distributed_inference.model_management.model_profile import profile_model
import onnx
import networkx as nx

onnx_model = onnx.load("./assets/models/resnet50.onnx")
model_graph = profile_model(onnx_model)

topo_sort = nx.topological_sort(model_graph)
for node in topo_sort:
    print(node)

for edge in model_graph.edges:
    print(edge, model_graph.get_edge_data(edge[0], edge[1]))
