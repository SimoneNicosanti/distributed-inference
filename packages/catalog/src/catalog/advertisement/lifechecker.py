from abc import ABC, abstractmethod


class Lifechecker(ABC):
    @abstractmethod
    async def check_life(self) -> bool: ...
