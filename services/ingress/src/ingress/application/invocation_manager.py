from abc import ABC, abstractmethod


class InvocationManager(ABC):
    @abstractmethod
    async def invoke(self, request_data: str) -> str: ...
