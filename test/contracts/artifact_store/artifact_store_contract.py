from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from distributed_inference.artifact_store.application.ports.outbound.artifact_store import (
    ArtifactStore,
)
from distributed_inference.artifact_store.domain.artifact_key import (
    ModelVersionArtifactKey,
    SubModelArtifactKey,
)
from distributed_inference.domain.identifiers import (
    UserId,
)
from distributed_inference.model_manager.domain.model import ModelId
from distributed_inference.model_manager.domain.model_version import ModelVersionId
from distributed_inference.model_manager.domain.sub_model import SubModelId
from test.support.artifact_store.artifact_bundle_test_utils import (
    build_test_bundle,
    read_bundle_content,
)


def _model_version_id(version_number: int = 1) -> ModelVersionId:
    return ModelVersionId(
        model_id=ModelId(
            owner_id=UserId(id=uuid4()),
            model_name="resnet50",
        ),
        version_tag=version_number,
    )


class ArtifactStoreContract(ABC):
    @abstractmethod
    def build_store(self, base_path: Path) -> ArtifactStore:
        raise NotImplementedError

    @pytest.fixture
    def store(self, tmp_path: Path) -> ArtifactStore:
        return self.build_store(tmp_path / "store")

    @pytest.mark.asyncio
    async def test_missing_artifact_does_not_exist(
        self,
        store: ArtifactStore,
    ) -> None:
        key = ModelVersionArtifactKey(id=_model_version_id())

        assert not await store.check_artifact_existence(key)

    @pytest.mark.asyncio
    async def test_put_artifact_preserves_manifest_and_file_contents(
        self,
        store: ArtifactStore,
        tmp_path: Path,
    ) -> None:
        key = ModelVersionArtifactKey(id=_model_version_id())
        bundle = build_test_bundle(
            tmp_path / "input",
            files={
                PurePosixPath("onnx/model.onnx"): b"model-content",
                PurePosixPath("onnx/weights/model.data"): b"weights-content",
            },
            entrypoint=PurePosixPath("onnx/model.onnx"),
        )

        await store.put_artifact(key, bundle)

        assert await store.check_artifact_existence(key)
        async with store.open_artifact(key) as stored:
            assert stored.manifest == bundle.manifest
            assert await read_bundle_content(stored) == {
                PurePosixPath("onnx/model.onnx"): b"model-content",
                PurePosixPath("onnx/weights/model.data"): b"weights-content",
            }

    @pytest.mark.asyncio
    async def test_open_missing_artifact_raises(
        self,
        store: ArtifactStore,
    ) -> None:
        key = ModelVersionArtifactKey(id=_model_version_id())

        with pytest.raises(FileNotFoundError):
            async with store.open_artifact(key):
                pass

    @pytest.mark.asyncio
    async def test_model_version_and_sub_model_keys_are_independent(
        self,
        store: ArtifactStore,
        tmp_path: Path,
    ) -> None:
        version_id = _model_version_id()
        version_key = ModelVersionArtifactKey(id=version_id)
        sub_model_key = SubModelArtifactKey(
            id=SubModelId(
                model_version_id=version_id,
                layers=("encoder.0",),
            )
        )
        version_bundle = build_test_bundle(
            tmp_path / "version",
            files={PurePosixPath("model.onnx"): b"whole-model"},
        )
        sub_model_bundle = build_test_bundle(
            tmp_path / "sub-model",
            files={PurePosixPath("model.onnx"): b"sub-model"},
        )

        await store.put_artifact(version_key, version_bundle)
        await store.put_artifact(sub_model_key, sub_model_bundle)

        async with store.open_artifact(version_key) as stored_version:
            version_content = await read_bundle_content(stored_version)
        async with store.open_artifact(sub_model_key) as stored_sub_model:
            sub_model_content = await read_bundle_content(stored_sub_model)

        assert version_content[PurePosixPath("model.onnx")] == b"whole-model"
        assert sub_model_content[PurePosixPath("model.onnx")] == b"sub-model"
