import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from worker.domain.profiling.model_execution.shape_point import ShapePoint


class Backend(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True, slots=True)
class LayerExecutionProfile:
    execution_time_s: float
    memory_bytes: int | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.execution_time_s) or self.execution_time_s < 0:
            raise ValueError("Execution time must be finite and non-negative")
        if self.memory_bytes is not None and (
            isinstance(self.memory_bytes, bool) or self.memory_bytes < 0
        ):
            raise ValueError("Memory must be a non-negative byte count")


@dataclass(frozen=True, slots=True)
class BackendExecutionProfile:
    backend: Backend
    layer_profiles: Mapping[LayerKey, LayerExecutionProfile]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_profiles",
            MappingProxyType(dict(self.layer_profiles)),
        )


@dataclass(frozen=True, slots=True)
class ShapeExecutionProfile:
    shape_point: ShapePoint
    backend_profiles: tuple[BackendExecutionProfile, ...]


@dataclass(frozen=True, slots=True)
class ModelExecutionProfile:
    model_version_id: ModelVariantId
    shape_execution_profiles: tuple[ShapeExecutionProfile, ...]
