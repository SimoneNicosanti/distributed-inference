from abc import ABC, abstractmethod
from typing import Any


class ResultCollector(ABC):
    @abstractmethod
    async def collect_result(
        self,
        request_id: Any,
        to_wait_list: list[Any],
    ) -> Any: ...

    @abstractmethod
    async def add_result(
        self,
        request_id: Any,
        result: Any,
    ) -> None: ...
