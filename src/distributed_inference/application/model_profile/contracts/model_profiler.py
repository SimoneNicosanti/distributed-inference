from abc import ABC, abstractmethod
from pathlib import Path

from distributed_inference.domain.model_graph_info import ModelGraph, ModelInfo


class ModelProfiler(ABC):
    @abstractmethod
    def profile_model(
        self,
        model_path: Path,
        model_info: ModelInfo,
    ) -> ModelGraph: ...
