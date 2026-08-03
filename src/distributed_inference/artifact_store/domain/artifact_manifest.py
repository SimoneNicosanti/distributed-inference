from pathlib import PurePosixPath
from typing import Any, Tuple, override

from pydantic import BaseModel, ConfigDict, field_validator

MANIFEST_FILE_NAME = "manifest.json"


def _validate_artifact_ppp(path: PurePosixPath) -> None:
    if path.is_absolute():
        raise ValueError(f"Artifact path must be relative: {path}")

    if path == PurePosixPath(".") or ".." in path.parts:
        raise ValueError(f"Invalid artifact path: {path}")

    if "\\" in path.as_posix():
        raise ValueError(f"Backslashes are not allowed: {path}")

    if path == PurePosixPath(MANIFEST_FILE_NAME):
        raise ValueError(f"{MANIFEST_FILE_NAME} is reserved for the bundle manifest")


class ArtifactFileInfo(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    file_ppp: PurePosixPath  ## Relatve path with respect to the bundle root

    @field_validator("file_ppp")
    @classmethod
    def validate_file_ppp(cls, value: PurePosixPath) -> PurePosixPath:
        _validate_artifact_ppp(value)

        return value


## Describes the structure of the model bundle with respect to the bundle root
class ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    entrypoint_ppp: PurePosixPath  ## Main file path with respect to bundle root (e.g., for onnx the model.onnx file)
    files_info: Tuple[
        ArtifactFileInfo, ...
    ]  ## ALL file paths with respect to bundle root

    def get_ppp_set(self) -> set[PurePosixPath]:
        return set([file_info.file_ppp for file_info in self.files_info])

    @override
    def model_post_init(
        self,
        __context: Any,
    ) -> None:

        ## Check that the entrypoint is valid
        _validate_artifact_ppp(self.entrypoint_ppp)

        ## Not repeated paths in the manifest declaration
        not_repeated_paths = set([file_info.file_ppp for file_info in self.files_info])
        if len(self.files_info) != len(not_repeated_paths):
            raise ValueError("Artifact manifest contains duplicate paths")

        ## Check if the entrypoint is declared as a file
        if self.entrypoint_ppp not in not_repeated_paths:
            raise ValueError("Artifact entrypoint is not declared in the manifest")
