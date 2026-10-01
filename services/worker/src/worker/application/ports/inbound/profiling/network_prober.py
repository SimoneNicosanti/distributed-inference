from abc import ABC, abstractmethod


class NetworkProber(ABC):
    @abstractmethod
    def profile_throughput(
        self,
    ) -> None: ...

    @abstractmethod
    def profile_round_trip_time(
        self,
    ) -> None: ...
