from typing import override

from artifacts.contracts.artifact_bundle import ArtifactBundle
from worker.application.activity.abc.activity_request_factory import (
    ActivityRequestFactory,
)
from worker.application.ports.outbound.activity_manager.activity_manager import (
    ActivityManager,
)
from worker.application.ports.outbound.profiling.model_execution.engine_model_execution_profiler import (
    EngineModelExecutionProfiler,
)
from worker.application.profiling.abc.model_execution_profiler import (
    ModelExecutionProfiler,
)
from worker.domain.activity.activity_request import ActivityType
from worker.domain.profiling.model_execution.model_execution_profile import (
    Backend,
    BackendExecutionProfile,
    ModelExecutionProfile,
    ShapeExecutionProfile,
)
from worker.domain.profiling.model_execution.model_static_profile import (
    ModelStaticProfile,
)


class DefaultModelExecutionProfiler(ModelExecutionProfiler):
    def __init__(
        self,
        activity_request_factory: ActivityRequestFactory,
        activity_manager: ActivityManager,
        backends: list[Backend],
        engine_profiler: EngineModelExecutionProfiler,
    ):
        self._activity_request_factory = activity_request_factory
        self._activity_manager = activity_manager

        if len(backends) == 0:
            raise ValueError("At least one backend must be provided")
        self._backends = backends

        self._engine_profiler = engine_profiler

    @override
    async def profile_model_execution(
        self, model_static_profile: ModelStaticProfile, artifact_bundle: ArtifactBundle
    ) -> ModelExecutionProfile:

        activity_request = (
            self._activity_request_factory.create_request_for_activity_type(
                ActivityType.EXECUTION_PROFILING
            )
        )
        activity_grant = await self._activity_manager.acquire_activity_grant(
            request=activity_request
        )
        async with activity_grant:
            shape_execution_profiles = await self._profile_model_on_all_shapes(
                model_static_profile, artifact_bundle
            )

            return ModelExecutionProfile(
                model_version_id=model_static_profile.model_variant_id,
                shape_execution_profiles=tuple(shape_execution_profiles),
            )

    async def _profile_model_on_all_shapes(
        self, model_static_profile: ModelStaticProfile, artifact_bundle: ArtifactBundle
    ) -> list[ShapeExecutionProfile]:
        shape_execution_profiles: list[ShapeExecutionProfile] = []
        for shape_point in model_static_profile.shape_points:
            backend_profiles: list[BackendExecutionProfile] = []
            for backend in self._backends:
                backend_profile = await self._engine_profiler.profile_model_execution_on_backend_with_shape_point(
                    model_static_profile, artifact_bundle, backend, shape_point
                )
                backend_profiles.append(backend_profile)

            shape_execution_profiles.append(
                ShapeExecutionProfile(
                    shape_point=shape_point,
                    backend_profiles=tuple(backend_profiles),
                )
            )
        return shape_execution_profiles
