from abc import ABC, abstractmethod

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.domain.model_graph_info import ModelGraph, ModelInfo


class ModelGraphExtractor(ABC):
    @abstractmethod
    def extract_model_graph(
        self,
        paths: ArtifactConcretePaths,
        model_info: ModelInfo,
        profile_flops: bool,
        profile_tensors: bool,
    ) -> ModelGraph: ...

    @abstractmethod
    def aggregate_model_graphs(
        self, level_1_graph: ModelGraph, level_2_graph: ModelGraph
    ) -> ModelGraph: ...
