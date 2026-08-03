from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from distributed_inference.artifact_store.domain.artifact_manifest import (
    ArtifactManifest,
)


## This is the path of the bundle root and of entrypoint as in the system
class MaterializedArtifact(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
    )
    manifest: ArtifactManifest
    root_path: Path
    entrypoint_path: Path

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if not self.root_path.is_dir():
            raise ValueError(f"Bundle root path is not a directory: {self.root_path}")

        if not self.entrypoint_path.is_file():
            raise ValueError(f"Entrypoint path is not a file: {self.entrypoint_path}")

        if not self.entrypoint_path.is_relative_to(self.root_path):
            raise ValueError("Entrypoint path must be contained inside the bundle root")

        if self.entrypoint_path != self.root_path.joinpath(
            self.manifest.entrypoint_ppp
        ):
            raise ValueError(
                "Entrypoint path does not match the manifest entrypoint path"
            )

        resolved_root = self.root_path.resolve(strict=True)
        resolved_entrypoint = self.entrypoint_path.resolve(strict=True)

        if not resolved_entrypoint.is_relative_to(resolved_root):
            raise ValueError("Entrypoint path must be contained inside the bundle root")

        return self
