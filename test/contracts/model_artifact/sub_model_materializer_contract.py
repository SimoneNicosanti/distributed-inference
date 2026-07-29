from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4

import pytest

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.model_artifact.application.ports.outbound.materializer.sub_model_materializer import (
    SubModelMaterializer,
)
from distributed_inference.model_artifact.application.ports.outbound.store.sub_model_artifact_store import (
    SubModelArtifactStore,
)
from test.support.model_artifact.artifact_bundle_test_utils import (
    build_test_bundle,
)
from test.support.model_artifact.materializer_test_utils import (
    extract_root_and_entrypoint,
)


def build_sub_model_id() -> SubModelId:
    model_id = ModelId(
        user_id=UserId(user_id=uuid4()),
        model_name="resnet50",
    )

    model_version_id = ModelVersionId(
        model_id=model_id,
        version_number=1,
    )

    return SubModelId(
        model_version_id=model_version_id,
        layers=(
            "layer_1",
            "layer_2",
        ),
    )


class SubModelMaterializerContract(ABC):
    @abstractmethod
    def build_dependencies(
        self,
        base_path: Path,
    ) -> tuple[
        SubModelMaterializer,
        SubModelArtifactStore,
    ]:
        raise NotImplementedError

    @pytest.fixture
    def dependencies(
        self,
        tmp_path: Path,
    ) -> tuple[
        SubModelMaterializer,
        SubModelArtifactStore,
    ]:
        return self.build_dependencies(tmp_path)

    @pytest.fixture
    def sub_model_id(self) -> SubModelId:
        return build_sub_model_id()

    def test_materialize_missing_sub_model_raises(
        self,
        dependencies: tuple[
            SubModelMaterializer,
            SubModelArtifactStore,
        ],
        sub_model_id: SubModelId,
    ) -> None:
        materializer, _ = dependencies

        with pytest.raises(FileNotFoundError):
            with materializer.materialize_sub_model(sub_model_id):
                pass

    def test_materialize_returns_root_and_entrypoint(
        self,
        dependencies: tuple[
            SubModelMaterializer,
            SubModelArtifactStore,
        ],
        sub_model_id: SubModelId,
    ) -> None:
        materializer, store = dependencies

        bundle = build_test_bundle(
            model_content=b"sub-model-content",
            weights_content=b"sub-model-weights",
        )

        store.put_sub_model(
            sub_model_id,
            bundle,
        )

        with materializer.materialize_sub_model(sub_model_id) as concrete_paths:
            root_path, entrypoint_path = extract_root_and_entrypoint(concrete_paths)

            expected_entrypoint = root_path.joinpath(
                *bundle.manifest.rel_entrypoint_path.parts
            )

            assert root_path.is_dir()
            assert entrypoint_path.is_file()
            assert entrypoint_path == expected_entrypoint
            assert entrypoint_path.is_relative_to(root_path)

    def test_materialized_root_contains_all_artifact_files(
        self,
        dependencies: tuple[
            SubModelMaterializer,
            SubModelArtifactStore,
        ],
        sub_model_id: SubModelId,
    ) -> None:
        materializer, store = dependencies

        bundle = build_test_bundle(
            model_content=b"sub-model-content",
            weights_content=b"sub-model-weights",
        )

        store.put_sub_model(
            sub_model_id,
            bundle,
        )

        with materializer.materialize_sub_model(sub_model_id) as concrete_paths:
            root_path, _ = extract_root_and_entrypoint(concrete_paths)

            model_path = root_path / "model.onnx"
            weights_path = root_path / "weights" / "model.data"

            assert model_path.read_bytes() == b"sub-model-content"
            assert weights_path.read_bytes() == b"sub-model-weights"
