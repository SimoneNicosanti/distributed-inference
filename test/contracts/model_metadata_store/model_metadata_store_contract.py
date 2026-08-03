from abc import ABC, abstractmethod
from uuid import uuid4

import pytest

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    SubModelId,
    UserId,
)
from distributed_inference.domain.model_graph_info import (
    LayerKey,
    ModelGraph,
    ModelInfo,
    ModelType,
    TaskType,
)
from distributed_inference.model_metadata_store.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
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
        return ModelInfo(
            name="resnet50",
            accuracy=0.9,
            task=TaskType.CLASSIFICATION,
            type=ModelType.CNN,
            dynamic_shapes={},
            sequence_sizes=[1],
            num_heads=0,
            hidden_size=0,
        )

    @pytest.fixture
    def model_graph(self, model_info: ModelInfo) -> ModelGraph:
        return ModelGraph(model_info=model_info)

    @pytest.fixture
    def layers(self) -> tuple[LayerKey, ...]:
        return (
            "layer_1",
            "layer_2",
        )

    @pytest.mark.asyncio
    async def test_registered_model_exists(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        model_id = await store.register_model(
            owner_id=owner_id,
            model_name="resnet50",
        )

        assert await store.check_model_existence(model_id)

    @pytest.mark.asyncio
    async def test_unregistered_model_does_not_exist(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        model_id = ModelId(
            user_id=owner_id,
            model_name="missing",
        )

        assert not await store.check_model_existence(model_id)

    @pytest.mark.asyncio
    async def test_registering_same_model_twice_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        await store.register_model(owner_id, "resnet50")

        with pytest.raises(ValueError, match="already exists"):
            await store.register_model(owner_id, "resnet50")

    @pytest.mark.asyncio
    async def test_same_model_name_is_allowed_for_different_owners(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        second_owner_id: UserId,
    ) -> None:
        first_model_id = await store.register_model(
            owner_id,
            "resnet50",
        )
        second_model_id = await store.register_model(
            second_owner_id,
            "resnet50",
        )

        assert first_model_id != second_model_id
        assert await store.check_model_existence(first_model_id)
        assert await store.check_model_existence(second_model_id)

    @pytest.mark.asyncio
    async def test_first_model_version_has_number_one(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")

        version_id = await store.register_model_version(
            model_id=model_id,
            model_info=model_info,
        )

        assert version_id.model_id == model_id
        assert version_id.version_number == 1
        assert await store.check_model_version_existence(version_id)

    @pytest.mark.asyncio
    async def test_model_version_numbers_increment(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")

        first = await store.register_model_version(model_id, model_info)
        second = await store.register_model_version(model_id, model_info)
        third = await store.register_model_version(model_id, model_info)

        assert first.version_number == 1
        assert second.version_number == 2
        assert third.version_number == 3

        assert first != second
        assert second != third

        assert await store.check_model_version_existence(first)
        assert await store.check_model_version_existence(second)
        assert await store.check_model_version_existence(third)

    @pytest.mark.asyncio
    async def test_version_numbers_are_independent_for_each_model(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        first_model_id = await store.register_model(
            owner_id,
            "resnet50",
        )
        second_model_id = await store.register_model(
            owner_id,
            "vit",
        )

        first_version = await store.register_model_version(
            first_model_id,
            model_info,
        )
        second_version = await store.register_model_version(
            second_model_id,
            model_info,
        )

        assert first_version.version_number == 1
        assert second_version.version_number == 1

    @pytest.mark.asyncio
    async def test_registering_version_for_missing_model_raises(
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
            await store.register_model_version(
                model_id=missing_model_id,
                model_info=model_info,
            )

    @pytest.mark.asyncio
    async def test_unregistered_model_version_does_not_exist(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")

        missing_version_id = ModelVersionId(
            model_id=model_id,
            version_number=999,
        )

        assert not await store.check_model_version_existence(missing_version_id)

    @pytest.mark.asyncio
    async def test_get_model_info_returns_registered_info(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")
        version_id = await store.register_model_version(
            model_id,
            model_info,
        )

        assert await store.get_model_info(version_id) == model_info

    @pytest.mark.asyncio
    async def test_get_model_info_for_missing_version_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")

        missing_version_id = ModelVersionId(
            model_id=model_id,
            version_number=1,
        )

        with pytest.raises(ValueError, match="Model version"):
            await store.get_model_info(missing_version_id)

    @pytest.mark.asyncio
    async def test_model_graph_is_initially_none(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")
        version_id = await store.register_model_version(
            model_id,
            model_info,
        )

        assert await store.get_model_graph(version_id) is None

    @pytest.mark.asyncio
    async def test_register_and_get_model_graph(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
        model_graph: ModelGraph,
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")
        version_id = await store.register_model_version(
            model_id,
            model_info,
        )

        await store.register_model_version_graph(
            model_version_id=version_id,
            model_graph=model_graph,
        )

        assert await store.get_model_graph(version_id) == model_graph

    @pytest.mark.asyncio
    async def test_registering_graph_for_missing_version_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_graph: ModelGraph,
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")

        missing_version_id = ModelVersionId(
            model_id=model_id,
            version_number=1,
        )

        with pytest.raises(ValueError, match="Model version"):
            await store.register_model_version_graph(
                model_version_id=missing_version_id,
                model_graph=model_graph,
            )

    @pytest.mark.asyncio
    async def test_get_graph_for_missing_version_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")

        missing_version_id = ModelVersionId(
            model_id=model_id,
            version_number=1,
        )

        with pytest.raises(ValueError, match="Model version"):
            await store.get_model_graph(missing_version_id)

    @pytest.mark.asyncio
    async def test_registered_submodel_exists(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
        layers: tuple[LayerKey, ...],
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")
        version_id = await store.register_model_version(
            model_id,
            model_info,
        )

        sub_model_id = await store.register_sub_model(
            model_version_id=version_id,
            layers=layers,
        )

        assert sub_model_id.model_version_id == version_id
        assert sub_model_id.layers == layers
        assert await store.check_sub_model_existence(sub_model_id)

    @pytest.mark.asyncio
    async def test_submodel_registration_is_idempotent(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
        layers: tuple[LayerKey, ...],
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")
        version_id = await store.register_model_version(
            model_id,
            model_info,
        )

        first = await store.register_sub_model(version_id, layers)
        second = await store.register_sub_model(version_id, layers)

        assert first == second
        assert await store.check_sub_model_existence(first)

    @pytest.mark.asyncio
    async def test_different_layers_produce_different_submodels(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")
        version_id = await store.register_model_version(
            model_id,
            model_info,
        )

        first = await store.register_sub_model(
            version_id,
            ("layer_1",),
        )
        second = await store.register_sub_model(
            version_id,
            ("layer_2",),
        )

        assert first != second
        assert await store.check_sub_model_existence(first)
        assert await store.check_sub_model_existence(second)

    @pytest.mark.asyncio
    async def test_registering_submodel_for_missing_version_raises(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        layers: tuple[LayerKey, ...],
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")

        missing_version_id = ModelVersionId(
            model_id=model_id,
            version_number=1,
        )

        with pytest.raises(ValueError, match="Model version"):
            await store.register_sub_model(
                model_version_id=missing_version_id,
                layers=layers,
            )

    @pytest.mark.asyncio
    async def test_unregistered_submodel_does_not_exist(
        self,
        store: ModelMetadataStore,
        owner_id: UserId,
        model_info: ModelInfo,
        layers: tuple[LayerKey, ...],
    ) -> None:
        model_id = await store.register_model(owner_id, "resnet50")
        version_id = await store.register_model_version(
            model_id,
            model_info,
        )

        sub_model_id = SubModelId(
            model_version_id=version_id,
            layers=layers,
        )

        assert not await store.check_sub_model_existence(sub_model_id)
