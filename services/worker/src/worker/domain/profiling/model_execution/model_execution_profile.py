import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from worker.domain.profiling.model_execution.shape_point import ShapePoint


@dataclass(frozen=True, slots=True)
class BackendLayerExecutionProfile:
    execution_time: float
    memory: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.execution_time) or self.execution_time < 0:
            raise ValueError("Execution time must be finite and non-negative")
        if not math.isfinite(self.memory) or self.memory < 0:
            raise ValueError("Memory must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LayerExecutionProfile:
    layer_key: LayerKey

    cpu_execution_profile: BackendLayerExecutionProfile
    gpu_execution_profile: BackendLayerExecutionProfile | None


@dataclass(frozen=True, slots=True)
class ShapeExecutionProfile:
    shape_point: ShapePoint
    layer_profiles: Mapping[LayerKey, LayerExecutionProfile]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_profiles",
            MappingProxyType(dict(self.layer_profiles)),
        )


@dataclass(frozen=True, slots=True)
class ModelExecutionProfile:
    model_version_id: ModelVariantId
    shape_execution_profiles: tuple[ShapeExecutionProfile, ...]
