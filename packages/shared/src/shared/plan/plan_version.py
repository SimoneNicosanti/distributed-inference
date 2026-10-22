from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class PlanVersion:
    version_number: int

    def __post_init__(self) -> None:
        if isinstance(self.version_number, bool) or self.version_number <= 0:
            raise ValueError("Plan version number must be positive")
