from abc import ABC, abstractmethod
from typing import Iterable

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
)
from distributed_inference.domain.model_graph_info import LayerKey, ModelGraph


class ModelSplitter(ABC):
    @abstractmethod
    async def split_model(
        self,
        model_graph: ModelGraph,
        layers: Iterable[LayerKey],
        input_paths: MaterializedArtifact,
        output_paths: ArtifactWorkspace,
    ) -> None: ...
