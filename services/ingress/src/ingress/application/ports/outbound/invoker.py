from abc import ABC, abstractmethod
from typing import Any


class Invoker(ABC):
    @abstractmethod
    async def invoke(
        self,
        request: Any,
    ) -> None: ...
