from abc import ABC, abstractmethod

from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
)
from distributed_inference.domain.model_graph_info import ModelGraph, ModelInfo


## NOTE: For now, we keep these calls synchronous since they are compute intensive
class ModelGraphExtractor(ABC):
    @abstractmethod
    def extract_model_graph(
        self,
        paths: ArtifactWorkspace,
        model_info: ModelInfo,
        profile_flops: bool,
        profile_tensors: bool,
    ) -> ModelGraph: ...

    @abstractmethod
    def aggregate_model_graphs(
        self, level_1_graph: ModelGraph, level_2_graph: ModelGraph
    ) -> ModelGraph: ...
