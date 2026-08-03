from pathlib import Path

from distributed_inference.model_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)


def extract_root_and_entrypoint(
    concrete_paths: MaterializedArtifact,
) -> tuple[Path, Path]:
    assert concrete_paths.entrypoint_path is not None
    return concrete_paths.root_path, concrete_paths.entrypoint_path
