from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from artifacts.contracts.artifact_manifest import ArtifactFileInfo, ArtifactManifest
from artifacts.contracts.artifact_workspace import ArtifactWorkspace


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

    def to_workspace(self) -> ArtifactWorkspace:
        return ArtifactWorkspace(
            root_path=self.root_path,
            entrypoint_path=self.root_path / self.manifest.entrypoint_ppp,
        )

    @classmethod
    def from_workspace(cls, workspace: ArtifactWorkspace) -> ArtifactBundle:
        if workspace.entrypoint_path is None:
            raise ValueError("Artifact workspace entrypoint path is not set")

        if not workspace.root_path.is_dir():
            raise ValueError(
                f"Artifact workspace root path is not a directory: "
                f"{workspace.root_path}"
            )

        if not workspace.entrypoint_path.is_file():
            raise ValueError(
                f"Artifact workspace entrypoint is not a file: "
                f"{workspace.entrypoint_path}"
            )

        resolved_root = workspace.root_path.resolve(strict=True)
        resolved_entrypoint = workspace.entrypoint_path.resolve(strict=True)
        if not resolved_entrypoint.is_relative_to(resolved_root):
            raise ValueError(
                f"Artifact workspace entrypoint is outside the workspace root: "
                f"{workspace.entrypoint_path}"
            )

        file_ppps = sorted(
            (
                PurePosixPath(*path.relative_to(workspace.root_path).parts)
                for path in workspace.root_path.rglob("*")
                if path.is_file()
            ),
            key=PurePosixPath.as_posix,
        )
        entrypoint_ppp = PurePosixPath(
            *resolved_entrypoint.relative_to(resolved_root).parts
        )

        return cls(
            manifest=ArtifactManifest(
                entrypoint_ppp=entrypoint_ppp,
                files_info=tuple(
                    ArtifactFileInfo(file_ppp=file_ppp) for file_ppp in file_ppps
                ),
            ),
            root_path=workspace.root_path,
        )
