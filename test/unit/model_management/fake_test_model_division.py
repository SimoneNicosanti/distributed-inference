from argparse import ArgumentParser, Namespace
from pathlib import Path
from models_info import get_model_info, BASE_MODEL_PATH


from distributed_inference.application.model_management.profiling.model_profile_agg import (
    compute_aggregate_model_graph,
    profile_with_model_optimization,
)


from distributed_inference.application.model_management.optimization.model_optimize import (
    OptimizationLevel,
)

from distributed_inference.application.model_management.division.model_division import (
    divide_model,
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

    topological_sort = agg_model_graph.get_topological_sort()

    sub_layers_1 = topological_sort[0:100]
    sub_layers_2 = topological_sort[100:200]
    sub_layers_3 = topological_sort[-100:]

    divide_model(
        agg_model_graph,
        set(sub_layers_1),
        model_path,
        Path("./outputs/model_divide/test_1.onnx"),
    )
    divide_model(
        agg_model_graph,
        set(sub_layers_2),
        model_path,
        Path("./outputs/model_divide/test_2.onnx"),
    )
    divide_model(
        agg_model_graph,
        set(sub_layers_3),
        model_path,
        Path("./outputs/model_divide/test_3.onnx"),
    )


if __name__ == "__main__":
    main()
