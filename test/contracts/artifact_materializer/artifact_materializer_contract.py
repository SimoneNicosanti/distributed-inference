from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from distributed_inference.artifact_materializer.application.ports.outbound.artifact_materializer import (
    ArtifactMaterializer,
)
from distributed_inference.artifact_store.application.ports.outbound.artifact_store import (
    ArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import (
    ModelVersionArtifactKey,
)
from distributed_inference.domain.identifiers import (
    UserId,
)
from distributed_inference.model_manager.domain.model import ModelId
from distributed_inference.model_manager.domain.model_version import ModelVersionId
from test.support.artifact_store.artifact_bundle_test_utils import build_test_bundle


class ArtifactMaterializerContract(ABC):
    @abstractmethod
    def build_materializer_and_store(
        self,
        base_path: Path,
    ) -> tuple[ArtifactMaterializer, ArtifactStore]:
        raise NotImplementedError

    @pytest.mark.asyncio
    async def test_materialize_exposes_manifest_root_and_entrypoint(
        self,
        tmp_path: Path,
    ) -> None:
        materializer, store = self.build_materializer_and_store(tmp_path / "store")
        key = ModelVersionArtifactKey(
            id=ModelVersionId(
                model_id=ModelId(
                    owner_id=UserId(id=uuid4()),
                    model_name="resnet50",
                ),
                version_tag="v1",
            )
        )
        bundle = build_test_bundle(
            tmp_path / "input",
            files={
                PurePosixPath("model/model.onnx"): b"model",
                PurePosixPath("model/weights.data"): b"weights",
            },
            entrypoint=PurePosixPath("model/model.onnx"),
        )
        await store.put_artifact(key, bundle)

        async with materializer.materialize_artifact(key) as materialized:
            assert materialized.manifest == bundle.manifest
            assert materialized.root_path.is_dir()
            assert materialized.entrypoint_path == (
                materialized.root_path / "model" / "model.onnx"
            )
            assert materialized.entrypoint_path.read_bytes() == b"model"
            assert (materialized.root_path / "model" / "weights.data").is_file()

    @pytest.mark.asyncio
    async def test_materialize_missing_artifact_raises(
        self,
        tmp_path: Path,
    ) -> None:
        materializer, _ = self.build_materializer_and_store(tmp_path / "store")
        key = ModelVersionArtifactKey(
            id=ModelVersionId(
                model_id=ModelId(
                    owner_id=UserId(id=uuid4()),
                    model_name="missing",
                ),
                version_tag="v1",
            )
        )

        with pytest.raises(FileNotFoundError):
            async with materializer.materialize_artifact(key):
                pass
