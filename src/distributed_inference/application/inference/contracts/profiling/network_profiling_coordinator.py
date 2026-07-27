from abc import ABC, abstractmethod


class NetworkProfilingCoordinator(ABC):
    @abstractmethod
    def profile_whole_network(self) -> None: ...

    @abstractmethod
    def profile_network_to_destination(self) -> None: ...
