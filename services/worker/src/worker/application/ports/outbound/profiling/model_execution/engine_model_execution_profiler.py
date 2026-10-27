from abc import ABC, abstractmethod

from artifacts.contracts.artifact_bundle import ArtifactBundle
from shared.model.model_variant import ModelVariantId
from worker.domain.profiling.model_execution.model_execution_profile import (
    Backend,
    BackendExecutionProfile,
)
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)
from worker.domain.profiling.model_execution.shape_point import ShapePoint


class EngineModelExecutionProfiler(ABC):
    @abstractmethod
    async def profile_model_execution_on_backend_with_shape_point(
        self,
        model_static_profile: ModelStaticProfile,
        artifact_bundle: ArtifactBundle,
        backend: Backend,
        shape_point: ShapePoint,
    ) -> BackendExecutionProfile: ...
