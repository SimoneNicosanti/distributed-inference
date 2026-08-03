from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
)
from distributed_inference.model_artifact.application.ports.outbound.model_version_artifact_store import (
    ModelVersionArtifactStore,
)
from distributed_inference.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactFile,
    ArtifactManifest,
)
from test.support.model_artifact.artifact_bundle_test_utils import (
    build_test_bundle,
    read_bundle_content,
)


class ModelVersionArtifactStoreContract(ABC):
    @abstractmethod
    def build_store(
        self,
        base_path: Path,
    ) -> ModelVersionArtifactStore:
        raise NotImplementedError

    @pytest.fixture
    def store(
        self,
        tmp_path: Path,
    ) -> ModelVersionArtifactStore:
        return self.build_store(tmp_path)

    @pytest.fixture
    def model_version_id(self) -> ModelVersionId:
        model_id = ModelId(
            user_id=UserId(user_id=uuid4()),
            model_name="resnet50",
        )

        return ModelVersionId(
            model_id=model_id,
            version_number=1,
        )

    def test_missing_model_version_does_not_exist(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        assert not store.check_model_version_existence(model_version_id)

    def test_put_model_version_makes_bundle_exist(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        store.put_model_version(
            model_version_id,
            build_test_bundle(),
        )

        assert store.check_model_version_existence(model_version_id)

    def test_get_model_version_preserves_manifest(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        input_bundle = build_test_bundle()

        store.put_model_version(
            model_version_id,
            input_bundle,
        )

        with store.get_model_version(model_version_id) as result:
            assert result.manifest == input_bundle.manifest

    def test_get_model_version_returns_all_file_contents(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        store.put_model_version(
            model_version_id,
            build_test_bundle(
                model_content=b"model-content",
                weights_content=b"weights-content",
            ),
        )

        with store.get_model_version(model_version_id) as result:
            contents = read_bundle_content(result)

        assert contents == {
            PurePosixPath("model.onnx"): b"model-content",
            PurePosixPath("weights/model.data"): b"weights-content",
        }

    def test_nested_relative_paths_are_preserved(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        bundle = ArtifactBundle(
            manifest=ArtifactManifest(
                rel_entrypoint_path=PurePosixPath("onnx/model.onnx"),
                rel_file_paths=(
                    PurePosixPath("onnx/model.onnx"),
                    PurePosixPath("onnx/weights/part_0.data"),
                    PurePosixPath("onnx/weights/part_1.data"),
                ),
            ),
            artifact_files=(
                ArtifactFile(
                    rel_path=PurePosixPath("onnx/model.onnx"),
                    content=BytesIO(b"model"),
                ),
                ArtifactFile(
                    rel_path=PurePosixPath("onnx/weights/part_0.data"),
                    content=BytesIO(b"part-zero"),
                ),
                ArtifactFile(
                    rel_path=PurePosixPath("onnx/weights/part_1.data"),
                    content=BytesIO(b"part-one"),
                ),
            ),
        )

        store.put_model_version(
            model_version_id,
            bundle,
        )

        with store.get_model_version(model_version_id) as result:
            contents = read_bundle_content(result)

        assert contents == {
            PurePosixPath("onnx/model.onnx"): b"model",
            PurePosixPath("onnx/weights/part_0.data"): b"part-zero",
            PurePosixPath("onnx/weights/part_1.data"): b"part-one",
        }

    def test_get_model_version_keeps_streams_open_inside_context(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        store.put_model_version(
            model_version_id,
            build_test_bundle(),
        )

        with store.get_model_version(model_version_id) as result:
            assert all(
                not artifact_file.content.closed
                for artifact_file in result.artifact_files
            )

    def test_get_model_version_closes_streams_after_context(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        store.put_model_version(
            model_version_id,
            build_test_bundle(),
        )

        with store.get_model_version(model_version_id) as result:
            streams = tuple(
                artifact_file.content for artifact_file in result.artifact_files
            )

        assert all(stream.closed for stream in streams)

    def test_get_missing_model_version_raises(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            with store.get_model_version(model_version_id):
                pass

    def test_different_versions_are_independent(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        second_version_id = ModelVersionId(
            model_id=model_version_id.model_id,
            version_number=2,
        )

        store.put_model_version(
            model_version_id,
            build_test_bundle(
                model_content=b"version-one",
                weights_content=b"weights-one",
            ),
        )
        store.put_model_version(
            second_version_id,
            build_test_bundle(
                model_content=b"version-two",
                weights_content=b"weights-two",
            ),
        )

        with store.get_model_version(model_version_id) as first:
            first_contents = read_bundle_content(first)

        with store.get_model_version(second_version_id) as second:
            second_contents = read_bundle_content(second)

        assert first_contents[PurePosixPath("model.onnx")] == (b"version-one")
        assert second_contents[PurePosixPath("model.onnx")] == (b"version-two")

    def test_different_models_are_independent(
        self,
        store: ModelVersionArtifactStore,
        model_version_id: ModelVersionId,
    ) -> None:
        second_model_id = ModelId(
            user_id=model_version_id.model_id.user_id,
            model_name="vit",
        )
        second_version_id = ModelVersionId(
            model_id=second_model_id,
            version_number=1,
        )

        store.put_model_version(
            model_version_id,
            build_test_bundle(model_content=b"resnet"),
        )
        store.put_model_version(
            second_version_id,
            build_test_bundle(model_content=b"vit"),
        )

        with store.get_model_version(model_version_id) as first:
            first_contents = read_bundle_content(first)

        with store.get_model_version(second_version_id) as second:
            second_contents = read_bundle_content(second)

        assert first_contents[PurePosixPath("model.onnx")] == b"resnet"
        assert second_contents[PurePosixPath("model.onnx")] == b"vit"
