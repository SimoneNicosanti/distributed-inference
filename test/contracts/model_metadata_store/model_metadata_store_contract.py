from abc import ABC, abstractmethod

import pytest

from distributed_inference.model_manager.domain.model_version import (
    ModelVersion,
    ModelVersionId,
)
from distributed_inference.model_metadata_store.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)
from test.support.model_manager.model_domain_test_utils import (
    build_model,
    build_model_id,
    build_model_version,
    build_model_version_id,
    build_profiled_model_version,
    build_sub_model,
    build_sub_model_id,
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

    ## Model APIs

    @pytest.mark.asyncio
    async def test_registered_model_is_retrievable(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model = build_model()

        model_id = await store.register_model(model)

        assert model_id == model.model_id
        assert await store.check_model_existence(model_id)
        assert await store.get_model(model_id) == model

    @pytest.mark.asyncio
    async def test_unregistered_model_does_not_exist(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model_id = build_model_id(model_name="missing")

        assert not await store.check_model_existence(model_id)
        with pytest.raises(ValueError, match="does not exist"):
            await store.get_model(model_id)

    @pytest.mark.asyncio
    async def test_registering_the_same_model_twice_raises(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model = build_model()
        await store.register_model(model)

        with pytest.raises(ValueError, match="already exists"):
            await store.register_model(model)

    @pytest.mark.asyncio
    async def test_same_model_name_is_allowed_for_different_owners(
        self,
        store: ModelMetadataStore,
    ) -> None:
        first_model = build_model(model_id=build_model_id(model_name="resnet50"))
        second_model = build_model(model_id=build_model_id(model_name="resnet50"))

        first_model_id = await store.register_model(first_model)
        second_model_id = await store.register_model(second_model)

        assert first_model_id != second_model_id
        assert await store.check_model_existence(first_model_id)
        assert await store.check_model_existence(second_model_id)

    ## Model Version APIs

    @pytest.mark.asyncio
    async def test_registered_model_version_is_retrievable(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model = build_model()
        await store.register_model(model)
        model_version = build_model_version(
            model_version_id=build_model_version_id(model_id=model.model_id)
        )

        model_version_id = await store.register_model_version(model_version)

        assert model_version_id == model_version.model_version_id
        assert await store.check_model_version_existence(model_version_id)
        assert await store.get_model_version(model_version_id) == model_version

    @pytest.mark.asyncio
    async def test_version_tags_are_independent_for_each_model(
        self,
        store: ModelMetadataStore,
    ) -> None:
        first_model = build_model(model_id=build_model_id(model_name="resnet50"))
        second_model = build_model(model_id=build_model_id(model_name="vit"))
        await store.register_model(first_model)
        await store.register_model(second_model)

        first_version_id = await store.register_model_version(
            build_model_version(
                model_version_id=build_model_version_id(
                    model_id=first_model.model_id,
                    version_tag="v1",
                )
            )
        )
        second_version_id = await store.register_model_version(
            build_model_version(
                model_version_id=build_model_version_id(
                    model_id=second_model.model_id,
                    version_tag="v1",
                )
            )
        )

        assert first_version_id != second_version_id
        assert await store.check_model_version_existence(first_version_id)
        assert await store.check_model_version_existence(second_version_id)

    @pytest.mark.asyncio
    async def test_registering_the_same_model_version_twice_raises(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model = build_model()
        await store.register_model(model)
        model_version = build_model_version(
            model_version_id=build_model_version_id(model_id=model.model_id)
        )
        await store.register_model_version(model_version)

        with pytest.raises(ValueError, match="already exists"):
            await store.register_model_version(model_version)

    @pytest.mark.asyncio
    async def test_registering_a_version_for_a_missing_model_raises(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model_version = build_model_version(
            model_version_id=build_model_version_id(
                model_id=build_model_id(model_name="missing")
            )
        )

        with pytest.raises(ValueError, match="does not exist"):
            await store.register_model_version(model_version)

    @pytest.mark.asyncio
    async def test_unregistered_model_version_does_not_exist(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model = build_model()
        await store.register_model(model)
        missing_version_id = build_model_version_id(
            model_id=model.model_id,
            version_tag="missing",
        )

        assert not await store.check_model_version_existence(missing_version_id)
        with pytest.raises(ValueError, match="Model version"):
            await store.get_model_version(missing_version_id)

    ## Profiled Model Version APIs

    @pytest.mark.asyncio
    async def test_profiled_model_version_is_initially_absent(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model_version = await self._register_model_version(store)

        assert not await store.check_profiled_model_version_existence(
            model_version.model_version_id
        )
        assert (
            await store.get_profiled_model_version(model_version.model_version_id)
            is None
        )

    @pytest.mark.asyncio
    async def test_registered_profiled_model_version_is_retrievable(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model_version = await self._register_model_version(store)
        profiled_model_version = build_profiled_model_version(
            model_version=model_version
        )

        model_version_id = await store.register_profiled_model_version(
            profiled_model_version
        )

        assert model_version_id == model_version.model_version_id
        assert await store.check_profiled_model_version_existence(model_version_id)
        assert (
            await store.get_profiled_model_version(model_version_id)
            == profiled_model_version
        )

    @pytest.mark.asyncio
    async def test_registering_a_profile_for_a_missing_version_raises(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model = build_model()
        await store.register_model(model)
        profiled_model_version = build_profiled_model_version(
            model_version=build_model_version(
                model_version_id=build_model_version_id(
                    model_id=model.model_id,
                    version_tag="missing",
                )
            )
        )

        with pytest.raises(ValueError, match="Model version"):
            await store.register_profiled_model_version(profiled_model_version)

    ## Sub Model APIs

    @pytest.mark.asyncio
    async def test_registered_sub_model_is_retrievable(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model_version_id = await self._register_profiled_model_version(store)
        sub_model = build_sub_model(
            sub_model_id=build_sub_model_id(model_version_id=model_version_id)
        )

        sub_model_id = await store.register_sub_model(sub_model)

        assert sub_model_id == sub_model.sub_model_id
        assert await store.check_sub_model_existence(sub_model_id)
        assert await store.get_sub_model(sub_model_id) == sub_model

    @pytest.mark.asyncio
    async def test_sub_model_registration_is_idempotent(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model_version_id = await self._register_profiled_model_version(store)
        sub_model = build_sub_model(
            sub_model_id=build_sub_model_id(model_version_id=model_version_id)
        )

        first = await store.register_sub_model(sub_model)
        second = await store.register_sub_model(sub_model)

        assert first == second
        assert await store.check_sub_model_existence(first)

    @pytest.mark.asyncio
    async def test_different_layers_produce_different_sub_models(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model_version_id = await self._register_profiled_model_version(store)

        first = await store.register_sub_model(
            build_sub_model(
                sub_model_id=build_sub_model_id(
                    model_version_id=model_version_id,
                    layers=("encoder.0",),
                )
            )
        )
        second = await store.register_sub_model(
            build_sub_model(
                sub_model_id=build_sub_model_id(
                    model_version_id=model_version_id,
                    layers=("encoder.1",),
                )
            )
        )

        assert first != second
        assert await store.check_sub_model_existence(first)
        assert await store.check_sub_model_existence(second)

    @pytest.mark.asyncio
    async def test_registering_a_sub_model_for_a_missing_version_raises(
        self,
        store: ModelMetadataStore,
    ) -> None:
        sub_model = build_sub_model(
            sub_model_id=build_sub_model_id(
                model_version_id=build_model_version_id(version_tag="missing")
            )
        )

        with pytest.raises(ValueError, match="Model version"):
            await store.register_sub_model(sub_model)

    @pytest.mark.asyncio
    async def test_unregistered_sub_model_does_not_exist(
        self,
        store: ModelMetadataStore,
    ) -> None:
        model_version_id = await self._register_profiled_model_version(store)
        sub_model_id = build_sub_model_id(model_version_id=model_version_id)

        assert not await store.check_sub_model_existence(sub_model_id)
        with pytest.raises(ValueError, match="Sub model"):
            await store.get_sub_model(sub_model_id)

    ## Helpers

    async def _register_model_version(self, store: ModelMetadataStore) -> ModelVersion:
        model = build_model()
        await store.register_model(model)
        model_version = build_model_version(
            model_version_id=build_model_version_id(model_id=model.model_id)
        )
        await store.register_model_version(model_version)
        return model_version

    async def _register_profiled_model_version(
        self, store: ModelMetadataStore
    ) -> ModelVersionId:
        model_version = await self._register_model_version(store)
        await store.register_profiled_model_version(
            build_profiled_model_version(model_version=model_version)
        )
        return model_version.model_version_id
