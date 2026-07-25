from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactConcretePaths,
    ArtifactFile,
    ArtifactManifest,
)


@contextmanager
def build_artifact_bundle_from_bundle_paths(
    bundle_paths: ArtifactConcretePaths,
) -> Generator[ArtifactBundle]:

    manifest = build_manifest_from_bundle_paths(bundle_paths)

    artifact_files: list[ArtifactFile] = []
    with ExitStack() as stack:
        for rel_file_path in manifest.rel_file_paths:
            file_path = bundle_paths.root_path.joinpath(*rel_file_path.parts)

            bundle_file_content = stack.enter_context(file_path.open("rb"))

            artifact_file = ArtifactFile(
                rel_path=rel_file_path,
                content=bundle_file_content,
            )
            artifact_files.append(artifact_file)

        yield ArtifactBundle(manifest=manifest, artifact_files=tuple(artifact_files))


def build_manifest_from_bundle_paths(
    bundle_paths: ArtifactConcretePaths,
) -> ArtifactManifest:
    root_path = bundle_paths.root_path
    entrypoint_path = bundle_paths.entrypoint_path

    assert entrypoint_path is not None

    rel_entrypoint_path = entrypoint_path.relative_to(root_path)
    rel_file_paths: list[Path] = []
    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue
        rel_file_paths.append(file_path.relative_to(root_path))

    manifest = ArtifactManifest(
        rel_entrypoint_path=PurePosixPath(*rel_entrypoint_path.parts),
        rel_file_paths=tuple(
            [PurePosixPath(*rel_file_path.parts) for rel_file_path in rel_file_paths]
        ),
    )

    return manifest
