from abc import ABC, abstractmethod


## This is the interface that the control plane uses to manage the lifecycle of the inference service
class InferenceServiceManager(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...
