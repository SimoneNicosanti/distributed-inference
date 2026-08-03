from abc import ABC, abstractmethod

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.domain.model_graph_info import ModelGraph, ModelInfo


class ModelProfiler(ABC):
    @abstractmethod
    async def profile_model(
        self,
        artifact_concrete_paths: MaterializedArtifact,
        model_info: ModelInfo,
    ) -> ModelGraph: ...
