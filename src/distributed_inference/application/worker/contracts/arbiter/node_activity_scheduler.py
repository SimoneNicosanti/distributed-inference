from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum, StrEnum, auto
from typing import AsyncContextManager


class NodeActivityType(StrEnum):
    INFERENCE = auto()
    NETWORK_PROFILING = auto()
    EXECUTION_PROFILING = auto()
    MODEL_LOADING = auto()
    MODEL_UNLOADING = auto()
    MODEL_MIGRATE = auto()


class NodeActivityPriority(IntEnum):
    LOW = 0
    LOW_MEDIUM = 10
    MEDIUM = 20
    MEDIUM_HIGH = 30
    HIGH = 40


@dataclass
class NodeActivityRequest:
    type: NodeActivityType
    priority: NodeActivityPriority


## Syncronous context acquisition
## The context manager is acquired immediately
## Then we will wait on the context manager for the permission release
class NodeActivityScheduler(ABC):
    @abstractmethod
    def acquire(
        self,
        request: NodeActivityRequest,
    ) -> AsyncContextManager[None]: ...
