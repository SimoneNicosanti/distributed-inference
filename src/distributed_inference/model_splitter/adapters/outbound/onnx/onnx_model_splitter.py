import asyncio
from pathlib import Path
from typing import Iterable, override

import aiofiles
import aiofiles.os
import aiofiles.ospath

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
)
from distributed_inference.model_manager.domain.model_version_graph import (
    LayerKey,
    ModelVersionGraph,
)
from distributed_inference.model_splitter.application.ports.outbound.model_splitter import (
    ModelSplitter,
)
from onnx.utils import extract_model


class OnnxModelSplitter(ModelSplitter):
    @override
    async def split_model(
        self,
        model_graph: ModelVersionGraph,
        layers: Iterable[LayerKey],
        input_paths: MaterializedArtifact,
        output_paths: ArtifactWorkspace,
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

        ## TODO: Might this extraction become long? In that case, we should run it in a to_thread
        component_inputs, component_outputs = (
            model_graph.extract_incoming_outgoing_tensors_of_sub_model(set(layers))
        )

        input_names = list(component_inputs)
        output_names = list(component_outputs)

        if not input_names:
            raise ValueError("The extracted component has no inputs")

        if not output_names:
            raise ValueError("The extracted component has no outputs")

        await aiofiles.os.makedirs(output_model_path.parent, exist_ok=True)

        await asyncio.to_thread(
            self.__extract_model_sync,
            input_model_path,
            output_model_path,
            input_names,
            output_names,
        )

        output_paths.entrypoint_path = output_model_path

    def __extract_model_sync(
        self,
        input_model_path: Path,
        output_model_path: Path,
        input_names: list[str],
        output_names: list[str],
    ) -> None:
        extract_model(
            input_path=input_model_path,
            output_path=output_model_path,
            input_names=input_names,
            output_names=output_names,
            check_model=True,
            infer_shapes=True,
        )
