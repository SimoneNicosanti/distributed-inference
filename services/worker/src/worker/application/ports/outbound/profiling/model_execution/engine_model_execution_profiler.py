from abc import ABC, abstractmethod

from artifacts.contracts.artifact_bundle import ArtifactBundle
from worker.domain.profiling.model_execution.model_execution_profile import (
    Backend,
    BackendExecutionProfile,
)
from worker.domain.profiling.model_execution.model_input_concrete_shapes import (
    ModelInputConcreteShapes,
)
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)


class ModelExecutionProfilingError(Exception):
    pass


class EngineModelExecutionProfiler(ABC):
    @abstractmethod
    async def profile_model_execution_on_backend_with_input_shapes(
        self,
        model_static_profile: ModelStaticProfile,
        artifact_bundle: ArtifactBundle,
        backend: Backend,
        input_shapes: ModelInputConcreteShapes,
    ) -> BackendExecutionProfile: ...
