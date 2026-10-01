from abc import ABC, abstractmethod
from collections.abc import Iterable

from artifacts.workspace.artifact_workspace import (
    ArtifactWorkspace,
)
from model_manager.domain.model_version_graph import ModelVersionGraph
from shared.model.keys import LayerKey


class ModelSplitter(ABC):
    @abstractmethod
    async def split_model(
        self,
        model_graph: ModelVersionGraph,
        layers: Iterable[LayerKey],
        input_paths: ArtifactWorkspace,
        output_paths: ArtifactWorkspace,
    ) -> None: ...
