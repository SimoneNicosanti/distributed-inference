from enum import StrEnum


class SolverStrategyKind(StrEnum):
    FULL = "full"
    INCREMENTAL = "incremental"
    INCREMENTAL_PERSISTENT = "incremental-persistent"
