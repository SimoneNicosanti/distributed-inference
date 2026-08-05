from abc import ABC, abstractmethod
from typing import Iterable

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


class ModelSplitter(ABC):
    @abstractmethod
    async def split_model(
        self,
        model_graph: ModelVersionGraph,
        layers: Iterable[LayerKey],
        input_paths: MaterializedArtifact,
        output_paths: ArtifactWorkspace,
    ) -> None: ...
