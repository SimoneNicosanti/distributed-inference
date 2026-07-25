from abc import ABC, abstractmethod
from typing import Iterable

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.domain.model_graph_info import LayerKey, ModelGraph


class ModelSplitter(ABC):
    @abstractmethod
    def split_model(
        self,
        model_graph: ModelGraph,
        layers: Iterable[LayerKey],
        input_paths: ArtifactConcretePaths,
        output_paths: ArtifactConcretePaths,
    ) -> None: ...
