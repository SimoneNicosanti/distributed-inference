from abc import ABC, abstractmethod

from shared.service.service import WorkerId
from shared.model.model_variant import ModelVariantId
from worker.domain.profiling.model_execution_profile import ModelExecutionProfile


class ModelExecutionProfileStore(ABC):
    @abstractmethod
    async def put_model_execution_profile(
        self, worker_id: WorkerId, model_execution_profile: ModelExecutionProfile
    ) -> None: ...

    @abstractmethod
    async def get_model_execution_profile_by_worker_id(
        self, worker_id: WorkerId, model_version_id: ModelVariantId
    ) -> ModelExecutionProfile: ...

    @abstractmethod
    async def get_all_profiles_by_worker_id(
        self, worker_id: WorkerId
    ) -> list[ModelExecutionProfile]: ...

    @abstractmethod
    async def get_all_profiled_model_ids_by_worker_id(
        self, worker_id: WorkerId
    ) -> list[ModelVariantId]: ...

    @abstractmethod
    async def check_model_execution_profile_exists(
        self, worker_id: WorkerId, model_version_id: ModelVariantId
    ) -> bool: ...
