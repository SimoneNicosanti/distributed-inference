from abc import ABC, abstractmethod

from distributed_inference.domain.model_graph_info import ModelGraph, ModelInfo
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)


class ModelProfiler(ABC):
    @abstractmethod
    def profile_model(
        self,
        artifact_concrete_paths: ArtifactConcretePaths,
        model_info: ModelInfo,
    ) -> ModelGraph: ...
