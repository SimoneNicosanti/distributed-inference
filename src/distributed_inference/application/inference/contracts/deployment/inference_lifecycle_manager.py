from abc import ABC, abstractmethod


## This is the interface that the control plane uses to manage the lifecycle of the inference worker
class InferenceLifecycleManager(ABC):
    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def unload(self) -> None: ...

    @abstractmethod
    async def migrate(self) -> None: ...
