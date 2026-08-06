from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from distributed_inference.artifact_store.adapters.outbound.local.local_readable_artifact_bundle import (
    LocalReadableArtifactBundle,
)
from distributed_inference.artifact_store.domain.artifact_manifest import (
    ArtifactFileInfo,
    ArtifactManifest,
)
from distributed_inference.artifact_store.domain.readable_artifact_bundle import (
    ReadableArtifactBundle,
)


def build_test_bundle(
    root_path: Path,
    *,
    files: Mapping[PurePosixPath, bytes] | None = None,
    entrypoint: PurePosixPath = PurePosixPath("model.onnx"),
) -> LocalReadableArtifactBundle:
    contents = files or {
        PurePosixPath("model.onnx"): b"onnx-model",
        PurePosixPath("weights/model.data"): b"external-weights",
    }

    for relative_path, content in contents.items():
        file_path = root_path.joinpath(*relative_path.parts)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    manifest = ArtifactManifest(
        entrypoint_ppp=entrypoint,
        files_info=tuple(ArtifactFileInfo(file_ppp=path) for path in contents),
    )
    return LocalReadableArtifactBundle(
        manifest=manifest,
        local_root_path=root_path,
    )


async def read_bundle_content(
    bundle: ReadableArtifactBundle,
) -> dict[PurePosixPath, bytes]:
    contents: dict[PurePosixPath, bytes] = {}
    for file_info in bundle.manifest.files_info:
        async with bundle.open_file(file_info.file_ppp) as reader:
            contents[file_info.file_ppp] = await reader.read()
    return contents
