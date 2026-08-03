from pathlib import Path, PurePosixPath

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.artifact_store.domain.artifact_manifest import (
    ArtifactFileInfo,
    ArtifactManifest,
)


def build_test_materialized_artifact(
    entrypoint_path: Path,
    *,
    root_path: Path | None = None,
) -> MaterializedArtifact:
    artifact_root = root_path or entrypoint_path.parent
    entrypoint_ppp = PurePosixPath(
        entrypoint_path.relative_to(artifact_root).as_posix()
    )
    manifest = ArtifactManifest(
        entrypoint_ppp=entrypoint_ppp,
        files_info=(ArtifactFileInfo(file_ppp=entrypoint_ppp),),
    )
    return MaterializedArtifact(
        manifest=manifest,
        root_path=artifact_root,
        entrypoint_path=entrypoint_path,
    )
