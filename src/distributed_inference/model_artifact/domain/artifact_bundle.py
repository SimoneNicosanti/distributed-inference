from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Self, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import override

MANIFEST_FILE_NAME = "manifest.json"


def validate_artifact_rel_path(path: PurePosixPath) -> None:
    if path.is_absolute():
        raise ValueError(f"Artifact path must be relative: {path}")

    if path == PurePosixPath(".") or ".." in path.parts:
        raise ValueError(f"Invalid artifact path: {path}")

    if "\\" in path.as_posix():
        raise ValueError(f"Backslashes are not allowed: {path}")

    if path == PurePosixPath(MANIFEST_FILE_NAME):
        raise ValueError(f"{MANIFEST_FILE_NAME} is reserved for the bundle manifest")


## This is the class representing the bundle of files making up the model
## - manifest: manifest of the bundle
## - artifact_files: files making up the bundle
## NOTE: We cannot model it as a BaseModel becuase of the BinaryIOs used in the ArtifactFile class
@dataclass(frozen=True)
class ArtifactBundle:
    manifest: ArtifactManifest
    artifact_files: Tuple[ArtifactFile, ...]

    def __post_init__(self) -> None:
        validate_artifact_rel_path(self.manifest.rel_entrypoint_path)

        declared_paths = self.manifest.rel_file_paths
        actual_paths = tuple(
            artifact_file.rel_path for artifact_file in self.artifact_files
        )

        for path in declared_paths:
            validate_artifact_rel_path(path)

        if len(set(declared_paths)) != len(declared_paths):
            raise ValueError("Artifact manifest contains duplicate paths")

        if len(set(actual_paths)) != len(actual_paths):
            raise ValueError("Artifact bundle contains duplicate files")

        if self.manifest.rel_entrypoint_path not in declared_paths:
            raise ValueError("Artifact entrypoint is not declared in the manifest")

        if set(declared_paths) != set(actual_paths):
            raise ValueError("Manifest paths do not match artifact files")


## File wrapper for model artifacts inside a bundle
## - rel_path: relative path with respect to the bundle root
## - content: file content
## We use this structure to handle models made up of multiple files
@dataclass(frozen=True)
class ArtifactFile:
    rel_path: PurePosixPath  ## Relatve path with respect to the bundle root
    content: BinaryIO

    def __post_init__(self) -> None:
        validate_artifact_rel_path(self.rel_path)

        ## TODO: Possiblle additional checks: check if the content is a real BinaryIO


## Describes the structure of the model bundle with respect to the bundle root
class ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    rel_entrypoint_path: PurePosixPath  ## Main file path with respect to bundle root (e.g., for onnx the model.onnx file)
    rel_file_paths: Tuple[
        PurePosixPath, ...
    ]  ## ALL file paths with respect to bundle root

    ## TODO To check other consistency information we should add other info to the manifest

    @override
    def model_post_init(
        self,
        __context: Any,
    ) -> None:

        ## Check that the entrypoint is valid
        validate_artifact_rel_path(self.rel_entrypoint_path)

        ## Check that all paths are valid relative paths
        for path in self.rel_file_paths:
            validate_artifact_rel_path(path)

        not_repeated_paths = set(self.rel_file_paths)

        if len(not_repeated_paths) != len(self.rel_file_paths):
            raise ValueError("Artifact manifest contains duplicate paths")

        ## Check if the entrypoint is declared as a file
        if self.rel_entrypoint_path not in not_repeated_paths:
            raise ValueError("Artifact entrypoint is not declared in the manifest")


## This is the path of the bundle root and of entrypoint as in the system
class ArtifactConcretePaths(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
    )

    root_path: Path = Field(frozen=True)
    entrypoint_path: Path | None = None

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if not self.root_path.exists():
            raise ValueError(f"Bundle root path does not exist: {self.root_path}")

        if not self.root_path.is_dir():
            raise ValueError(f"Bundle root path is not a directory: {self.root_path}")

        if self.entrypoint_path is None:
            return self

        if not self.entrypoint_path.exists():
            raise ValueError(f"Entrypoint path does not exist: {self.entrypoint_path}")

        if not self.entrypoint_path.is_file():
            raise ValueError(f"Entrypoint path is not a file: {self.entrypoint_path}")

        resolved_root = self.root_path.resolve(strict=True)
        resolved_entrypoint = self.entrypoint_path.resolve(strict=True)

        if not resolved_entrypoint.is_relative_to(resolved_root):
            raise ValueError("Entrypoint path must be contained inside the bundle root")

        return self
