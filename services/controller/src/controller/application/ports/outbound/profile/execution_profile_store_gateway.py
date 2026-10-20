from abc import ABC, abstractmethod
from collections.abc import Iterable

from controller.domain.profile.model_variant_execution_profile import (
    WorkerExecutionProfiles,
)
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class ExecutionProfileStoreGateway(ABC):
    @abstractmethod
    async def get_execution_profiles_for_workers_and_variants(
        self,
        worker_ids: Iterable[WorkerId],
        model_variant_ids: Iterable[ModelVariantId],
    ) -> list[WorkerExecutionProfiles]: ...
