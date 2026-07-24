from distributed_inference.application.model_management.profiling.model_profile import (
    profile_model,
)
from distributed_inference.domain.model_graph_info import (
    ModelGraph,
)
from argparse import ArgumentParser, Namespace
from pathlib import Path
import onnx
from models_info import get_model_info, BASE_MODEL_PATH


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    model_name = args.model
    model_path = BASE_MODEL_PATH / f"{model_name}.onnx"

    model_proto = onnx.load(str(model_path))
    model_info = get_model_info(model_name)

    model_profile = profile_model(model_proto, model_info, True)
    assert model_profile is not None

    for layer, layer_info in model_profile.get_all_layers().items():
        print(layer_info)

    for edge, edge_info in model_profile.get_all_edges().items():
        print(edge_info)

    for tensor, tensor_info in model_profile.get_tensors_map().items():
        print(tensor_info)


if __name__ == "__main__":
    main()
