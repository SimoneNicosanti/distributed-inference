from argparse import ArgumentParser, Namespace
from pathlib import Path
from model_management.models_info import get_model_info, BASE_MODEL_PATH


from distributed_inference.application.model_management.profiling.model_profile_agg import (
    compute_aggregate_model_graph,
    profile_with_model_optimization,
)


from distributed_inference.application.model_management.optimization.model_optimize import (
    OptimizationLevel,
)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    model_name = args.model
    model_path = BASE_MODEL_PATH / f"{model_name}.onnx"

    # model_proto = onnx.load(str(model_path))
    model_info = get_model_info(model_name)

    base_model_graph = profile_with_model_optimization(
        model_path, model_info, OptimizationLevel.BASIC
    )

    ext_model_graph = profile_with_model_optimization(
        model_path, model_info, OptimizationLevel.EXTENDED
    )

    agg_model_graph = compute_aggregate_model_graph(base_model_graph, ext_model_graph)

    print("Numero di nodi Base:", len(base_model_graph.get_all_layers()))
    print("Numero di archi Base:", len(base_model_graph.get_all_edges()))

    print("Numero di nodi Extended:", len(ext_model_graph.get_all_layers()))
    print("Numero di archi Extended:", len(ext_model_graph.get_all_edges()))

    print("Numero di nodi Aggregate:", len(agg_model_graph.get_all_layers()))
    print("Numero di archi Aggregate:", len(agg_model_graph.get_all_edges()))


if __name__ == "__main__":
    main()
