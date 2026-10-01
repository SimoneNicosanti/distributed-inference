from abc import ABC, abstractmethod


class Deployer(ABC):
    @abstractmethod
    def deploy(
        self,
    ) -> None: ...
