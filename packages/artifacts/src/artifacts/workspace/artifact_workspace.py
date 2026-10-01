from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from artifacts.contracts.artifact_manifest import (
    ArtifactFileInfo,
    ArtifactManifest,
)
from artifacts.workspace.artifact_bundle import ArtifactBundle


@dataclass(frozen=False)
class ArtifactWorkspace:
    root_path: Path
    entrypoint_path: Path | None = None


def build_artifact_workspace_from_artifact_bundle(
    artifact_bundle: ArtifactBundle,
) -> ArtifactWorkspace:
    return ArtifactWorkspace(
        root_path=artifact_bundle.root_path,
        entrypoint_path=artifact_bundle.root_path
        / artifact_bundle.manifest.entrypoint_ppp,
    )


def build_artifact_bundle_from_artifact_workspace(
    artifact_workspace: ArtifactWorkspace,
) -> ArtifactBundle:

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

    return ArtifactBundle(
        manifest=manifest,
        root_path=root_path,
    )


def build_artifact_bundle_from_root_path_and_manifest(
    root_path: Path,
    manifest: ArtifactManifest,
) -> ArtifactBundle:
    return ArtifactBundle(
        manifest=manifest,
        root_path=root_path,
    )
