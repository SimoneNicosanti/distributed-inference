from abc import ABC, abstractmethod


class StaticPriorityAssigner[ReqT](ABC):
    @abstractmethod
    def assign_priority(self, request: ReqT) -> int: ...
