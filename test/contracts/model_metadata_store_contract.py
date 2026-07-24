from abc import ABC, abstractmethod
from typing import cast
from unittest.mock import Mock
from uuid import uuid4

import pytest

from distributed_inference.application.model_metadata_store.contracts.model_metadata_store import (
    ModelMetadataStore,
)
from distributed_inference.domain.identifiers import (
    LayerKey,
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.domain.model_graph_info import (
    ModelGraph,
    ModelInfo,
)


class ModelMetadataStoreContract(ABC):
    """
    Test comportamentali comuni a tutte le implementazioni
    di ModelMetadataStore.

    Non viene raccolta direttamente da pytest perché il nome
    della classe non inizia con "Test".
    """

    @abstractmethod
    def build_store(self) -> ModelMetadataStore:
        raise NotImplementedError

    @pytest.fixture
    def store(self) -> ModelMetadataStore:
        # Ogni test riceve uno store pulito.
        return self.build_store()

    @pytest.fixture
    def owner_id(self) -> UserId:
        return UserId(user_id=uuid4())

    @pytest.fixture
    def second_owner_id(self) -> UserId:
        return UserId(user_id=uuid4())

    @pytest.fixture
    def model_info(self) -> ModelInfo:
        # Questi test non verificano la validazione di ModelInfo.
        return ModelInfo.model_construct(
            name="test-model",
            accuracy=0.9,
            dynamic_shapes={},
            sequence_sizes=[1],
            num_heads=0,
            hidden_size=0,
        )

    @pytest.fixture
    def model_graph(self) -> ModelGraph:
        # Per backend che serializzano i dati, questa fixture
        # potrà essere sovrascritta usando un ModelGraph reale.
        return cast(ModelGraph, Mock(spec=ModelGraph))

    @pytest.fixture
    def layers(self) -> tuple[LayerKey, ...]:
        return (
            "layer_1",
            "layer_2",
        )

    def test_registered_model_exists(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        model_id = store.register_model(
            owner_id=owner_id,
            model_name="resnet50",
        )

        assert store.check_model_existence(model_id)

    def test_unregistered_model_does_not_exist(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        model_id = ModelId(
            user_id=owner_id,
            model_name="missing",
        )

        assert not store.check_model_existence(model_id)

    def test_registering_same_model_twice_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        store.register_model(owner_id, "resnet50")

        with pytest.raises(ValueError, match="already exists"):
            store.register_model(owner_id, "resnet50")

    def test_same_model_name_is_allowed_for_different_owners(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        second_owner_id: UserId,
    ) -> None:
        first_model_id = store.register_model(
            owner_id,
            "resnet50",
        )
        second_model_id = store.register_model(
            second_owner_id,
            "resnet50",
        )

        assert first_model_id != second_model_id
        assert store.check_model_existence(first_model_id)
        assert store.check_model_existence(second_model_id)

    def test_first_model_version_has_number_one(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")

        version_id = store.register_model_version(
            model_id=model_id,
            model_info=model_info,
        )

        assert version_id.model_id == model_id
        assert version_id.version_number == 1
        assert store.check_model_version_existence(version_id)

    def test_model_version_numbers_increment(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")

        first = store.register_model_version(model_id, model_info)
        second = store.register_model_version(model_id, model_info)
        third = store.register_model_version(model_id, model_info)

        assert first.version_number == 1
        assert second.version_number == 2
        assert third.version_number == 3

        assert first != second
        assert second != third

        assert store.check_model_version_existence(first)
        assert store.check_model_version_existence(second)
        assert store.check_model_version_existence(third)

    def test_version_numbers_are_independent_for_each_model(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        first_model_id = store.register_model(
            owner_id,
            "resnet50",
        )
        second_model_id = store.register_model(
            owner_id,
            "vit",
        )

        first_version = store.register_model_version(
            first_model_id,
            model_info,
        )
        second_version = store.register_model_version(
            second_model_id,
            model_info,
        )

        assert first_version.version_number == 1
        assert second_version.version_number == 1

    def test_registering_version_for_missing_model_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        missing_model_id = ModelId(
            user_id=owner_id,
            model_name="missing",
        )

        with pytest.raises(ValueError, match="does not exist"):
            store.register_model_version(
                model_id=missing_model_id,
                model_info=model_info,
            )

    def test_unregistered_model_version_does_not_exist(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")

        missing_version_id = ModelVersionId(
            model_id=model_id,
            version_number=999,
        )

        assert not store.check_model_version_existence(missing_version_id)

    def test_get_model_info_returns_registered_info(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")
        version_id = store.register_model_version(
            model_id,
            model_info,
        )

        assert store.get_model_info(version_id) == model_info

    def test_get_model_info_for_missing_version_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")

        missing_version_id = ModelVersionId(
            model_id=model_id,
            version_number=1,
        )

        with pytest.raises(ValueError, match="Model version"):
            store.get_model_info(missing_version_id)

    def test_model_graph_is_initially_none(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")
        version_id = store.register_model_version(
            model_id,
            model_info,
        )

        assert store.get_model_graph(version_id) is None

    def test_register_and_get_model_graph(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
        model_graph: ModelGraph,
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")
        version_id = store.register_model_version(
            model_id,
            model_info,
        )

        store.register_model_version_graph(
            model_version_id=version_id,
            model_graph=model_graph,
        )

        assert store.get_model_graph(version_id) == model_graph

    def test_registering_graph_for_missing_version_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_graph: ModelGraph,
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")

        missing_version_id = ModelVersionId(
            model_id=model_id,
            version_number=1,
        )

        with pytest.raises(ValueError, match="Model version"):
            store.register_model_version_graph(
                model_version_id=missing_version_id,
                model_graph=model_graph,
            )

    def test_get_graph_for_missing_version_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")

        missing_version_id = ModelVersionId(
            model_id=model_id,
            version_number=1,
        )

        with pytest.raises(ValueError, match="Model version"):
            store.get_model_graph(missing_version_id)

    def test_registered_submodel_exists(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
        layers: tuple[LayerKey, ...],
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")
        version_id = store.register_model_version(
            model_id,
            model_info,
        )

        sub_model_id = store.register_sub_model(
            model_version_id=version_id,
            layers=layers,
        )

        assert sub_model_id.model_version_id == version_id
        assert sub_model_id.layers == layers
        assert store.check_sub_model_existence(sub_model_id)

    def test_submodel_registration_is_idempotent(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
        layers: tuple[LayerKey, ...],
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")
        version_id = store.register_model_version(
            model_id,
            model_info,
        )

        first = store.register_sub_model(version_id, layers)
        second = store.register_sub_model(version_id, layers)

        assert first == second
        assert store.check_sub_model_existence(first)

    def test_different_layers_produce_different_submodels(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")
        version_id = store.register_model_version(
            model_id,
            model_info,
        )

        first = store.register_sub_model(
            version_id,
            ("layer_1",),
        )
        second = store.register_sub_model(
            version_id,
            ("layer_2",),
        )

        assert first != second
        assert store.check_sub_model_existence(first)
        assert store.check_sub_model_existence(second)

    def test_registering_submodel_for_missing_version_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        layers: tuple[LayerKey, ...],
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")

        missing_version_id = ModelVersionId(
            model_id=model_id,
            version_number=1,
        )

        with pytest.raises(ValueError, match="Model version"):
            store.register_sub_model(
                model_version_id=missing_version_id,
                layers=layers,
            )

    def test_unregistered_submodel_does_not_exist(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
        layers: tuple[LayerKey, ...],
    ) -> None:
        model_id = store.register_model(owner_id, "resnet50")
        version_id = store.register_model_version(
            model_id,
            model_info,
        )

        sub_model_id = SubModelId(
            model_version_id=version_id,
            layers=layers,
        )

        assert not store.check_sub_model_existence(sub_model_id)
