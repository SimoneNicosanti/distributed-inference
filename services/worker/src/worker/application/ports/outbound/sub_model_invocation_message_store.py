from abc import ABC, abstractmethod

from worker.application.forwarding.contracts.gathering.gather_key import (
    GatherKey,
)
from worker.domain.sub_model.invocation.sub_model_invocation_message import (
    SubModelInvocationMessage,
)


class SubModelInvocationMessageGatheringStore(ABC):
    @abstractmethod
    async def put_sub_model_invocation_message(
        self, sub_model_invocation_message: SubModelInvocationMessage
    ) -> GatherKey: ...

    @abstractmethod
    async def get_by_gathering_key(
        self, gathering_key: GatherKey
    ) -> list[SubModelInvocationMessage]: ...

    @abstractmethod
    async def delete_by_gathering_key(self, gathering_key: GatherKey) -> None: ...
