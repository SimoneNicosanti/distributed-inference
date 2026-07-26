from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    MANIFEST_FILE_NAME,
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
    root_path = bundle_paths.root_path.resolve(strict=True)

    if bundle_paths.entrypoint_path is None:
        raise ValueError(
            "Bundle entrypoint path must be set when building manifest from bundle paths"
        )
    entrypoint_path = bundle_paths.entrypoint_path.resolve(strict=True)

    rel_entrypoint_path = entrypoint_path.relative_to(root_path)
    rel_file_paths: list[Path] = []
    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            ## We skip directories
            continue
        if file_path.relative_to(root_path) == Path(MANIFEST_FILE_NAME):
            ## We skip the manifest
            continue
        if file_path.is_symlink():
            ## We skip symlinks
            continue
        rel_file_paths.append(file_path.relative_to(root_path))

    manifest = ArtifactManifest(
        rel_entrypoint_path=PurePosixPath(*rel_entrypoint_path.parts),
        rel_file_paths=tuple(
            [PurePosixPath(*rel_file_path.parts) for rel_file_path in rel_file_paths]
        ),
    )

    return manifest


@contextmanager
def build_bundle_from_root_path_and_manifest(
    root_path: Path, manifest: ArtifactManifest
) -> Generator[ArtifactBundle]:

    entrypoint_path = root_path.joinpath(*manifest.rel_entrypoint_path.parts)
    if not entrypoint_path.is_file():
        raise ValueError("Declared entrypoint is not a file")

    artifact_files: list[ArtifactFile] = []
    with ExitStack() as stack:
        for rel_file_path in manifest.rel_file_paths:
            file_path = root_path.joinpath(*rel_file_path.parts)
            if not file_path.is_file():
                raise ValueError(
                    f"Declared file {rel_file_path} is not a file in the bundle"
                )
            if file_path.is_symlink():
                raise ValueError(
                    f"Declared file {rel_file_path} is a symlink in the bundle"
                )

            bundle_file_content = stack.enter_context(file_path.open("rb"))

            artifact_file = ArtifactFile(
                rel_path=rel_file_path,
                content=bundle_file_content,
            )
            artifact_files.append(artifact_file)

        artifact_bundle = ArtifactBundle(
            manifest=manifest, artifact_files=tuple(artifact_files)
        )

        yield artifact_bundle
