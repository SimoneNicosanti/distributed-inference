from abc import ABC, abstractmethod
from typing import Any


class ResultReceiver(ABC):
    @abstractmethod
    async def receive_result(self, request_id: Any, result_data: Any) -> None: ...
