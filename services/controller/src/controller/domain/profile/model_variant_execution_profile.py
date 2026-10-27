import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from controller.domain.profile.model_variant_static_profile import ShapePoint
from shared.model.keys import LayerKey
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class Backend(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True, slots=True)
class LayerExecutionProfile:
    execution_time_s: float
    memory_bytes: int | None = None

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
class ModelVariantExecutionProfile:
    shape_execution_profiles: tuple[ShapeExecutionProfile, ...]


@dataclass(frozen=True, slots=True)
class WorkerExecutionProfiles:
    worker_id: WorkerId
    execution_profiles: Mapping[ModelVariantId, ModelVariantExecutionProfile]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_profiles",
            MappingProxyType(dict(self.execution_profiles)),
        )
