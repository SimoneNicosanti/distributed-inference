from enum import StrEnum, auto


class CompactObjectiveKind(StrEnum):
    ARRIVAL_WEIGHTED_RESPONSE_TIME = auto()
    ARRIVAL_WEIGHTED_ACCURACY = auto()
    DEADLINE_VIOLATIONS = auto()
    ARRIVAL_WEIGHTED_NORMALIZED_DELAY = auto()
    CONSTANT = auto()
