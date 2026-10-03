from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=False)
class ArtifactWorkspace:
    root_path: Path
    entrypoint_path: Path | None = None
