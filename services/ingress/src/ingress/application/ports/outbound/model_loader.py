from abc import ABC, abstractmethod


class ModelUploader(ABC):
    @abstractmethod
    def upload(
        self,
    ) -> None: ...
