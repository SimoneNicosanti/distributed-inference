from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from artifacts.contracts.artifact_manifest import ArtifactManifest


class ArtifactBundle(BaseModel):
    """Validated local filesystem source for storing or archiving an artifact."""

    model_config = ConfigDict(frozen=True)

    manifest: ArtifactManifest
    root_path: Path

    @property
    def entrypoint_path(self) -> Path:
        return self.root_path.joinpath(*self.manifest.entrypoint_ppp.parts)

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        if not self.root_path.is_dir():
            raise ValueError(
                f"Artifact source root path is not a directory: {self.root_path}"
            )

        resolved_root = self.root_path.resolve(strict=True)
        for file_info in self.manifest.files_info:
            file_path = self.root_path.joinpath(*file_info.file_ppp.parts)
            if not file_path.is_file():
                raise ValueError(f"Artifact source file does not exist: {file_path}")

            if not file_path.resolve(strict=True).is_relative_to(resolved_root):
                raise ValueError(
                    f"Artifact source file is outside the source root: {file_path}"
                )

        return self
