from abc import ABC, abstractmethod


class RequestRouter(ABC):
    @abstractmethod
    def route_request(self): ...
