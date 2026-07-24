from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from distributed_inference.domain.model_graph_info import LayerKey, ModelGraph


class ModelSplitter(ABC):
    @abstractmethod
    def split_model(
        self,
        model_graph: ModelGraph,
        layers: Iterable[LayerKey],
        input_path: Path,
        output_path: Path,
    ) -> None: ...
