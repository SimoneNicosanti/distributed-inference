from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from distributed_inference.application.model_artifact.contracts.store.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactBundle,
    ArtifactFile,
    ArtifactManifest,
)
from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.domain.model_graph_info import LayerKey
from test.support.artifact_bundle_test_utils import (
    build_test_bundle,
    read_bundle_content,
)


def build_layers(*names: str) -> tuple[LayerKey, ...]:
    return tuple(name for name in names)


def build_sub_model_id(
    *,
    version_number: int = 1,
    layers: tuple[LayerKey, ...] | None = None,
    model_name: str = "resnet50",
) -> SubModelId:
    model_id = ModelId(
        user_id=UserId(user_id=uuid4()),
        model_name=model_name,
    )
    model_version_id = ModelVersionId(
        model_id=model_id,
        version_number=version_number,
    )

    return SubModelId(
        model_version_id=model_version_id,
        layers=layers or build_layers("layer_1", "layer_2"),
    )


class SubModelArtifactStoreContract(ABC):
    @abstractmethod
    def build_store(
        self,
        base_path: Path,
    ) -> SubModelArtifactStore:
        raise NotImplementedError

    @pytest.fixture
    def store(
        self,
        tmp_path: Path,
    ) -> SubModelArtifactStore:
        return self.build_store(tmp_path)

    @pytest.fixture
    def sub_model_id(self) -> SubModelId:
        return build_sub_model_id()

    def test_missing_sub_model_does_not_exist(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        assert not store.check_sub_model_existence(sub_model_id)

    def test_put_sub_model_makes_bundle_exist(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        store.put_sub_model(
            sub_model_id,
            build_test_bundle(),
        )

        assert store.check_sub_model_existence(sub_model_id)

    def test_get_sub_model_preserves_manifest(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        input_bundle = build_test_bundle()

        store.put_sub_model(
            sub_model_id,
            input_bundle,
        )

        with store.get_sub_model(sub_model_id) as result:
            assert result.manifest == input_bundle.manifest

    def test_get_sub_model_returns_all_file_contents(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        store.put_sub_model(
            sub_model_id,
            build_test_bundle(
                model_content=b"sub-model",
                weights_content=b"sub-model-weights",
            ),
        )

        with store.get_sub_model(sub_model_id) as result:
            contents = read_bundle_content(result)

        assert contents == {
            PurePosixPath("model.onnx"): b"sub-model",
            PurePosixPath("weights/model.data"): b"sub-model-weights",
        }

    def test_nested_relative_paths_are_preserved(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        entrypoint = PurePosixPath("onnx/sub_model.onnx")
        first_weights = PurePosixPath("onnx/weights/part_0.data")
        second_weights = PurePosixPath("onnx/weights/part_1.data")

        bundle = ArtifactBundle(
            manifest=ArtifactManifest(
                rel_entrypoint_path=entrypoint,
                rel_file_paths=(
                    entrypoint,
                    first_weights,
                    second_weights,
                ),
            ),
            artifact_files=(
                ArtifactFile(
                    rel_path=entrypoint,
                    content=BytesIO(b"sub-model"),
                ),
                ArtifactFile(
                    rel_path=first_weights,
                    content=BytesIO(b"part-zero"),
                ),
                ArtifactFile(
                    rel_path=second_weights,
                    content=BytesIO(b"part-one"),
                ),
            ),
        )

        store.put_sub_model(sub_model_id, bundle)

        with store.get_sub_model(sub_model_id) as result:
            contents = read_bundle_content(result)

        assert contents == {
            entrypoint: b"sub-model",
            first_weights: b"part-zero",
            second_weights: b"part-one",
        }

    def test_get_sub_model_keeps_streams_open_inside_context(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        store.put_sub_model(
            sub_model_id,
            build_test_bundle(),
        )

        with store.get_sub_model(sub_model_id) as result:
            assert all(
                not artifact_file.content.closed
                for artifact_file in result.artifact_files
            )

    def test_get_sub_model_closes_streams_after_context(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        store.put_sub_model(
            sub_model_id,
            build_test_bundle(),
        )

        with store.get_sub_model(sub_model_id) as result:
            streams = tuple(
                artifact_file.content for artifact_file in result.artifact_files
            )

        assert all(stream.closed for stream in streams)

    def test_get_missing_sub_model_raises(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            with store.get_sub_model(sub_model_id):
                pass

    def test_different_layer_sets_are_independent(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        second_sub_model_id = SubModelId(
            model_version_id=sub_model_id.model_version_id,
            layers=build_layers("layer_3", "layer_4"),
        )

        store.put_sub_model(
            sub_model_id,
            build_test_bundle(model_content=b"first-sub-model"),
        )
        store.put_sub_model(
            second_sub_model_id,
            build_test_bundle(model_content=b"second-sub-model"),
        )

        with store.get_sub_model(sub_model_id) as first_bundle:
            first_contents = read_bundle_content(first_bundle)

        with store.get_sub_model(second_sub_model_id) as second_bundle:
            second_contents = read_bundle_content(second_bundle)

        assert first_contents[PurePosixPath("model.onnx")] == (b"first-sub-model")
        assert second_contents[PurePosixPath("model.onnx")] == (b"second-sub-model")

    def test_different_model_versions_are_independent(
        self,
        store: SubModelArtifactStore,
        sub_model_id: SubModelId,
    ) -> None:
        second_version_id = ModelVersionId(
            model_id=sub_model_id.model_version_id.model_id,
            version_number=2,
        )
        second_sub_model_id = SubModelId(
            model_version_id=second_version_id,
            layers=sub_model_id.layers,
        )

        store.put_sub_model(
            sub_model_id,
            build_test_bundle(model_content=b"version-one"),
        )
        store.put_sub_model(
            second_sub_model_id,
            build_test_bundle(model_content=b"version-two"),
        )

        with store.get_sub_model(sub_model_id) as first_bundle:
            first_contents = read_bundle_content(first_bundle)

        with store.get_sub_model(second_sub_model_id) as second_bundle:
            second_contents = read_bundle_content(second_bundle)

        assert first_contents[PurePosixPath("model.onnx")] == b"version-one"
        assert second_contents[PurePosixPath("model.onnx")] == b"version-two"
