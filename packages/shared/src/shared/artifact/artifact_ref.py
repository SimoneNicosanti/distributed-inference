from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Opaque identifier assigned by the artifact owner."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("Artifact reference must not be empty")
