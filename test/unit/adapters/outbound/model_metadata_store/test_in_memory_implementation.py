from uuid import uuid4

import pytest

# Adatta soltanto questo import al percorso reale.
from distributed_inference.adapters.outbound.model_metadata_store.in_memory import (
    InMemoryModelMetadataStore,
)
from distributed_inference.domain.identifiers import (
    UserId,
)
from distributed_inference.domain.model_graph_info import ModelInfo


@pytest.fixture
def store() -> InMemoryModelMetadataStore:
    return InMemoryModelMetadataStore()


@pytest.fixture
def owner_id() -> UserId:
    return UserId(user_id=uuid4())


@pytest.fixture
def model_info() -> ModelInfo:
    return ModelInfo.model_construct(
        name="test-model",
        accuracy=0.9,
        dynamic_shapes={},
        sequence_sizes=[1],
        num_heads=0,
        hidden_size=0,
    )


def test_internal_dictionaries_are_initially_empty(
    store: InMemoryModelMetadataStore,
) -> None:
    assert store._model_metadata == {}
    assert store._model_version_metadata == {}
    assert store.sub_model_metadata == {}


def test_register_model_stores_metadata_in_model_dictionary(
    store: InMemoryModelMetadataStore,
    owner_id: UserId,
) -> None:
    model_id = store.register_model(
        owner_id=owner_id,
        model_name="resnet50",
    )

    assert store._model_metadata.get(model_id, None) is not None

    metadata = store._model_metadata[model_id]

    assert metadata.model_id == model_id
    assert metadata.owner_id == owner_id
    assert metadata.name == "resnet50"


def test_register_version_stores_metadata_in_version_dictionary(
    store: InMemoryModelMetadataStore,
    owner_id: UserId,
    model_info: ModelInfo,
) -> None:
    model_id = store.register_model(owner_id, "resnet50")

    version_id = store.register_model_version(
        model_id=model_id,
        model_info=model_info,
    )

    assert store._model_version_metadata.get(version_id, None) is not None

    metadata = store._model_version_metadata[version_id]

    assert metadata.model_id == model_id
    assert metadata.model_version_id == version_id
    assert metadata.version_number == 1
    assert metadata.model_info == model_info
    assert metadata.model_graph is None


def test_register_submodel_stores_metadata_in_submodel_dictionary(
    store: InMemoryModelMetadataStore,
    owner_id: UserId,
    model_info: ModelInfo,
) -> None:
    model_id = store.register_model(owner_id, "resnet50")
    version_id = store.register_model_version(
        model_id,
        model_info,
    )

    sub_model_id = store.register_sub_model(
        model_version_id=version_id,
        layers=(
            "layer_1",
            "layer_2",
        ),
    )

    assert store.sub_model_metadata.get(sub_model_id, None) is not None

    metadata = store.sub_model_metadata[sub_model_id]

    assert metadata.sub_model_id == sub_model_id


def test_store_instances_have_independent_state(
    owner_id: UserId,
) -> None:
    first_store = InMemoryModelMetadataStore()
    second_store = InMemoryModelMetadataStore()

    model_id = first_store.register_model(
        owner_id=owner_id,
        model_name="resnet50",
    )

    assert model_id in first_store._model_metadata
    assert model_id not in second_store._model_metadata

    assert second_store._model_metadata == {}
    assert second_store._model_version_metadata == {}
    assert second_store.sub_model_metadata == {}


def test_store_instances_have_different_locks() -> None:
    first_store = InMemoryModelMetadataStore()
    second_store = InMemoryModelMetadataStore()

    assert first_store.lock is not second_store.lock


def test_version_existence_checks_parent_model_dictionary(
    store: InMemoryModelMetadataStore,
    owner_id: UserId,
    model_info: ModelInfo,
) -> None:
    model_id = store.register_model(owner_id, "resnet50")
    version_id = store.register_model_version(
        model_id,
        model_info,
    )

    # Corruzione intenzionale dello stato interno:
    # il record della versione resta presente, ma il modello padre no.
    del store._model_metadata[model_id]

    assert version_id in store._model_version_metadata
    assert not store.check_model_version_existence(version_id)
