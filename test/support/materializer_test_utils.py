from pathlib import Path

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)


def extract_root_and_entrypoint(
    concrete_paths: ArtifactConcretePaths,
) -> tuple[Path, Path]:
    assert concrete_paths.entrypoint_path is not None
    return concrete_paths.root_path, concrete_paths.entrypoint_path
