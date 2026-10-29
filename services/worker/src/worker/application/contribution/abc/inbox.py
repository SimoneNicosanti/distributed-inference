from abc import ABC, abstractmethod


class Inbox[KeyT, ValT](ABC):
    @abstractmethod
    async def extract_by_key(self, key: KeyT) -> ValT: ...

    @abstractmethod
    async def extract_next(
        self,
    ) -> ValT: ...
