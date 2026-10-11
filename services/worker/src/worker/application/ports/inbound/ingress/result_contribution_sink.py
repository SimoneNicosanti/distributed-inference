from abc import ABC, abstractmethod

from worker.domain.model_pass.model_pass_error import ModelPassError
from worker.domain.model_pass.model_pass_result_contribution import (
    ModelPassResultContribution,
)


class ResultContributionSink(ABC):
    @abstractmethod
    async def sink_contribution(
        self,
        contribution: ModelPassResultContribution,
    ) -> None: ...

    @abstractmethod
    async def notify_error(self, error: ModelPassError) -> None: ...
