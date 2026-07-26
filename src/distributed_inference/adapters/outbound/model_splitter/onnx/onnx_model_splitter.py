from typing import Iterable, override

from onnx.utils import extract_model

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.application.model_splitter.contracts.model_splitter import (
    ModelSplitter,
)
from distributed_inference.domain.model_graph_info import LayerKey, ModelGraph


class OnnxModelSplitter(ModelSplitter):
    @override
    def split_model(
        self,
        model_graph: ModelGraph,
        layers: Iterable[LayerKey],
        input_paths: ArtifactConcretePaths,
        output_paths: ArtifactConcretePaths,
    ) -> None:

        if not layers:
            raise ValueError("The component cannot be empty")

        if input_paths.entrypoint_path is None:
            raise ValueError("Entrypoint path must be set when splitting model")
        input_model_path = input_paths.entrypoint_path.resolve(strict=True)
        output_model_path = output_paths.root_path.resolve().joinpath(
            "split_model.onnx"
        )

        if input_model_path == output_model_path:
            raise ValueError("Input and output paths must be different")

        component_inputs, component_outputs = (
            model_graph.extract_incoming_outgoing_tensors_of_sub_model(set(layers))
        )

        input_names = list(component_inputs)
        output_names = list(component_outputs)

        if not input_names:
            raise ValueError("The extracted component has no inputs")

        if not output_names:
            raise ValueError("The extracted component has no outputs")

        output_model_path.parent.mkdir(parents=True, exist_ok=True)

        extract_model(
            input_path=input_model_path,
            output_path=output_model_path,
            input_names=input_names,
            output_names=output_names,
            check_model=True,
            infer_shapes=True,
        )

        output_paths.entrypoint_path = output_model_path
