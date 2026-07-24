from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from distributed_inference.domain.model_graph_info import ModelInfo


class OptimizationLevel(Enum):
    NONE = 0
    BASIC = 1
    EXTENDED = 2


class ModelOptimizer(ABC):
    @abstractmethod
    def optimize_model(
        self,
        input_path: Path,
        output_path: Path,
        model_info: ModelInfo,
        optimization_level: OptimizationLevel,
    ) -> None: ...
