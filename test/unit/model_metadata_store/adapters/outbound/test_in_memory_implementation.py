import pytest

from distributed_inference.model_metadata_store.adapters.outbound.in_memory_model_metadata_store import (
    InMemoryModelMetadataStore,
)
from test.support.model_manager.model_domain_test_utils import (
    build_model,
    build_model_version,
    build_model_version_id,
    build_profiled_model_version,
    build_sub_model,
    build_sub_model_id,
)


@pytest.fixture
def store() -> InMemoryModelMetadataStore:
    return InMemoryModelMetadataStore()


@pytest.mark.unit
def test_internal_dictionaries_are_initially_empty(
    store: InMemoryModelMetadataStore,
) -> None:
    assert store._models == {}
    assert store._model_versions == {}
    assert store._profiled_model_versions == {}
    assert store._sub_models == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registered_entities_are_stored_in_their_own_dictionary(
    store: InMemoryModelMetadataStore,
) -> None:
    model = build_model()
    await store.register_model(model)
    model_version = build_model_version(
        model_version_id=build_model_version_id(model_id=model.model_id)
    )
    await store.register_model_version(model_version)
    profiled_model_version = build_profiled_model_version(model_version=model_version)
    await store.register_profiled_model_version(profiled_model_version)
    sub_model = build_sub_model(
        sub_model_id=build_sub_model_id(model_version_id=model_version.model_version_id)
    )
    await store.register_sub_model(sub_model)

    assert store._models[model.model_id] is model
    assert store._model_versions[model_version.model_version_id] is model_version
    assert (
        store._profiled_model_versions[model_version.model_version_id]
        is profiled_model_version
    )
    assert store._sub_models[sub_model.sub_model_id] is sub_model


@pytest.mark.unit
@pytest.mark.asyncio
async def test_store_instances_have_independent_state() -> None:
    first_store = InMemoryModelMetadataStore()
    second_store = InMemoryModelMetadataStore()
    model = build_model()

    model_id = await first_store.register_model(model)

    assert model_id in first_store._models
    assert model_id not in second_store._models


@pytest.mark.unit
@pytest.mark.asyncio
async def test_profiled_model_version_registration_keeps_the_first_profile(
    store: InMemoryModelMetadataStore,
) -> None:
    model = build_model()
    await store.register_model(model)
    model_version = build_model_version(
        model_version_id=build_model_version_id(model_id=model.model_id)
    )
    await store.register_model_version(model_version)
    first_profile = build_profiled_model_version(model_version=model_version)
    second_profile = build_profiled_model_version(model_version=model_version)

    await store.register_profiled_model_version(first_profile)
    await store.register_profiled_model_version(second_profile)

    assert (
        store._profiled_model_versions[model_version.model_version_id] is first_profile
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sub_model_registration_requires_a_profiled_model_version(
    store: InMemoryModelMetadataStore,
) -> None:
    model = build_model()
    await store.register_model(model)
    model_version = build_model_version(
        model_version_id=build_model_version_id(model_id=model.model_id)
    )
    await store.register_model_version(model_version)
    sub_model = build_sub_model(
        sub_model_id=build_sub_model_id(model_version_id=model_version.model_version_id)
    )

    with pytest.raises(ValueError, match="Model version"):
        await store.register_sub_model(sub_model)
