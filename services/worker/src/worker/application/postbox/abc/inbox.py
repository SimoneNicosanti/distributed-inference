from abc import ABC, abstractmethod


class KeyedInbox[KeyT, ValT](ABC):
    @abstractmethod
    async def extract_by_key(self, key: KeyT) -> ValT: ...


class OrderedInbox[ValT](ABC):
    @abstractmethod
    async def extract_next(self) -> ValT: ...
