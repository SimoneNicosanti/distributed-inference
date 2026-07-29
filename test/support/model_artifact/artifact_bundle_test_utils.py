from io import BytesIO
from pathlib import PurePosixPath

from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactFile,
    ArtifactManifest,
)


def build_test_bundle(
    *,
    model_content: bytes = b"onnx-model",
    weights_content: bytes = b"external-weights",
) -> ArtifactBundle:
    model_path = PurePosixPath("model.onnx")
    weights_path = PurePosixPath("weights/model.data")

    return ArtifactBundle(
        manifest=ArtifactManifest(
            rel_entrypoint_path=model_path,
            rel_file_paths=(
                model_path,
                weights_path,
            ),
        ),
        artifact_files=(
            ArtifactFile(
                rel_path=model_path,
                content=BytesIO(model_content),
            ),
            ArtifactFile(
                rel_path=weights_path,
                content=BytesIO(weights_content),
            ),
        ),
    )


def read_bundle_content(
    bundle: ArtifactBundle,
) -> dict[PurePosixPath, bytes]:
    return {
        artifact_file.rel_path: artifact_file.content.read()
        for artifact_file in bundle.artifact_files
    }
