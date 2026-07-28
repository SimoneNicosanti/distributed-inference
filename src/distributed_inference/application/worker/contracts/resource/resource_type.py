from enum import StrEnum, auto


class ResourceType(StrEnum):
    COMPUTE = auto()
    MEMORY = auto()
    NETWORK = auto()