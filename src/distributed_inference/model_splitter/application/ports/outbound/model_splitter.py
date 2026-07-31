from abc import ABC, abstractmethod
from typing import Iterable

from distributed_inference.domain.model_graph_info import LayerKey, ModelGraph
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)


class ModelSplitter(ABC):
    @abstractmethod
    async def split_model(
        self,
        model_graph: ModelGraph,
        layers: Iterable[LayerKey],
        input_paths: ArtifactConcretePaths,
        output_paths: ArtifactConcretePaths,
    ) -> None: ...
