from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from distributed_inference.artifact_store.adapters.outbound.local.local_readable_artifact_bundle import (
    LocalReadableArtifactBundle,
)
from distributed_inference.artifact_store.domain.artifact_manifest import (
    ArtifactFileInfo,
    ArtifactManifest,
)


@dataclass(frozen=False)
class ArtifactWorkspace:
    root_path: Path
    entrypoint_path: Path | None = None


def build_local_artifact_bundle_from_artifact_workspace(
    artifact_workspace: ArtifactWorkspace,
) -> LocalReadableArtifactBundle:

    root_path = artifact_workspace.root_path
    if artifact_workspace.entrypoint_path is None:
        raise ValueError("Entrypoint path must be set to build the artifact bundle")
    entrypoint_path = artifact_workspace.entrypoint_path

    ## TODO: This might block the main executor
    entrypoint_ppp = entrypoint_path.relative_to(root_path)
    files_info: list[ArtifactFileInfo] = []
    file_paths = sorted(
        (path for path in root_path.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root_path).as_posix(),
    )

    for file_path in file_paths:
        file_ppp = file_path.relative_to(root_path)

        files_info.append(
            ArtifactFileInfo(
                file_ppp=PurePosixPath(file_ppp.as_posix()),
            )
        )
    manifest = ArtifactManifest(
        entrypoint_ppp=PurePosixPath(entrypoint_ppp.as_posix()),
        files_info=tuple(files_info),
    )

    return LocalReadableArtifactBundle(
        manifest=manifest,
        local_root_path=root_path,
    )


def build_local_artifact_bundle_from_root_path_and_manifest(
    root_path: Path,
    manifest: ArtifactManifest,
) -> LocalReadableArtifactBundle:

    for file_info in manifest.files_info:
        file_ppp = file_info.file_ppp
        file_path = root_path.joinpath(*file_ppp.parts)
        if not file_path.is_file():
            raise ValueError(f"File does not exist: {file_path}")

    return LocalReadableArtifactBundle(
        manifest=manifest,
        local_root_path=root_path,
    )
