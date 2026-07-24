from pathlib import Path
from typing import Iterable, override

from onnx.utils import extract_model

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
        input_path: Path,
        output_path: Path,
    ) -> None:

        if not layers:
            raise ValueError("The component cannot be empty")

        input_path = input_path.resolve(strict=True)
        output_path = output_path.resolve()

        if input_path == output_path:
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

        output_path.parent.mkdir(parents=True, exist_ok=True)

        extract_model(
            input_path=input_path,
            output_path=output_path,
            input_names=input_names,
            output_names=output_names,
            check_model=True,
            infer_shapes=True,
        )
