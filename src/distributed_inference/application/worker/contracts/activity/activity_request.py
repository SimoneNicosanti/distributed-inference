from dataclasses import dataclass
from enum import StrEnum, auto

from distributed_inference.application.worker.contracts.resource.resource_type import (
    ResourceType,
)


class ActivityType(StrEnum):
    INFERENCE_EXECUTION = auto()
    INFERENCE_FORWARDING = auto()

    PROFILING_NETWORK = auto()
    PROFILING_EXECUTION = auto()

    MODEL_LOADING = auto()
    MODEL_UNLOADING = auto()


@dataclass
class ActivityRequest:
    type: ActivityType
    resource_type: ResourceType


@dataclass
class ActivityResponse:
    response: bool
