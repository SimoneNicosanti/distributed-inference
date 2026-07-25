from pathlib import Path, PurePosixPath
from typing import BinaryIO, Tuple

from pydantic import BaseModel, ConfigDict


class ArtifactBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    MANIFEST_FILE_NAME = "manifest.json"

    manifest: ArtifactManifest
    artifact_files: Tuple[ArtifactFile, ...]


## File wrapper for model artifacts inside a bundle
## - rel_path: relative path with respect to the bundle root
## - content: file content
## We use this structure to handle models made up of multiple files
class ArtifactFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    rel_path: PurePosixPath  ## Relatve path with respect to the bundle root
    content: BinaryIO


## Describes the structure of the model bundle with respect to the bundle root
class ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    rel_entrypoint_path: PurePosixPath  ## Main file path with respect to bundle root (e.g., for onnx the model.onnx file)
    rel_file_paths: Tuple[
        PurePosixPath, ...
    ]  ## ALL file paths with respect to bundle root


## This is the path of the bundle root and of entrypoint as in the system
class ArtifactConcretePaths(BaseModel):
    model_config = ConfigDict(frozen=False)

    root_path: Path
    entrypoint_path: Path | None = None
