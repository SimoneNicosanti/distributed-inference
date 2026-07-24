from distributed_inference.application.model_metadata_store.contracts.model_metadata_store import (
    ModelMetadataStore,
)

from distributed_inference.domain.identifiers import (
    ModelId,
    ModelVersionId,
    UserId,
    SubModelId,
)

from distributed_inference.application.model_metadata_store.domain.model_metadata import (
    ModelMetadata,
    ModelVersionMetadata,
    SubModelMetadata,
)

from distributed_inference.domain.model_graph_info import (
    ModelInfo,
    ModelGraph,
    LayerKey,
)

from threading import Lock

from typing import Iterable

from typing import override


class InMemoryModelMetadataStore(ModelMetadataStore):
    def __init__(self) -> None:
        self.lock = Lock()
        self._model_metadata: dict[ModelId, ModelMetadata] = {}
        self._model_version_metadata: dict[ModelVersionId, ModelVersionMetadata] = {}
        self.sub_model_metadata: dict[SubModelId, SubModelMetadata] = {}

    @override
    def register_model(
        self,
        owner_id: UserId,
        model_name: str,
    ) -> ModelId:

        model_id = ModelId(user_id=owner_id, model_name=model_name)
        model_metdata = ModelMetadata(
            owner_id=owner_id,
            model_id=model_id,
            name=model_name,
        )

        with self.lock:
            if model_id in self._model_metadata.keys():
                ## We do not allow registering the same model twice
                raise ValueError(f"Model {model_id} already exists")
            self._model_metadata[model_id] = model_metdata

        return model_id

    @override
    def register_model_version(
        self,
        model_id: ModelId,
        model_info: ModelInfo,
    ) -> ModelVersionId:

        with self.lock:
            if model_id not in self._model_metadata.keys():
                raise ValueError(f"Model {model_id} does not exist")

            current_version_number = 0
            for model_version_metadata in self._model_version_metadata.values():
                if model_version_metadata.model_id == model_id:
                    current_version_number = max(
                        current_version_number,
                        model_version_metadata.version_number,
                    )

            model_version_id = ModelVersionId(
                model_id=model_id, version_number=current_version_number + 1
            )

            self._model_version_metadata[model_version_id] = ModelVersionMetadata(
                model_id=model_id,
                model_version_id=model_version_id,
                version_number=current_version_number + 1,
                model_info=model_info,
            )

        return model_version_id

    @override
    def register_model_version_graph(
        self,
        model_version_id: ModelVersionId,
        model_graph: ModelGraph,
    ) -> None:

        with self.lock:
            if model_version_id.model_id not in self._model_metadata.keys():
                raise ValueError(f"Model {model_version_id.model_id} does not exist")
            if model_version_id not in self._model_version_metadata.keys():
                raise ValueError(f"Model version {model_version_id} does not exist")
            self._model_version_metadata[model_version_id].model_graph = model_graph

    @override
    def register_sub_model(
        self,
        model_version_id: ModelVersionId,
        layers: Iterable[LayerKey],
    ) -> SubModelId:
        sub_model_id = SubModelId(
            model_version_id=model_version_id,
            layers=tuple(layers),
        )

        sub_model_metdata = SubModelMetadata(
            sub_model_id=sub_model_id,
        )

        with self.lock:
            if model_version_id not in self._model_version_metadata.keys():
                raise ValueError(f"Model version {model_version_id} does not exist")
            if model_version_id.model_id not in self._model_metadata.keys():
                raise ValueError(f"Model {model_version_id.model_id} does not exist")

            if sub_model_id in self.sub_model_metadata.keys():
                ## Idempotence
                return sub_model_id

            self.sub_model_metadata[sub_model_id] = sub_model_metdata

        return sub_model_id

    @override
    def get_model_graph(self, model_version_id: ModelVersionId) -> ModelGraph | None:
        with self.lock:
            if model_version_id.model_id not in self._model_metadata.keys():
                raise ValueError(f"Model {model_version_id.model_id} does not exist")
            if model_version_id not in self._model_version_metadata.keys():
                raise ValueError(f"Model version {model_version_id} does not exist")
            return self._model_version_metadata[model_version_id].model_graph

    @override
    def get_model_info(self, model_version_id: ModelVersionId) -> ModelInfo:
        with self.lock:
            if model_version_id.model_id not in self._model_metadata.keys():
                raise ValueError(f"Model {model_version_id.model_id} does not exist")
            if model_version_id not in self._model_version_metadata.keys():
                raise ValueError(f"Model version {model_version_id} does not exist")
            return self._model_version_metadata[model_version_id].model_info

    @override
    def check_model_existence(self, model_id: ModelId) -> bool:
        with self.lock:
            return model_id in self._model_metadata.keys()

    @override
    def check_model_version_existence(
        self,
        model_version_id: ModelVersionId,
    ) -> bool:
        with self.lock:
            return (
                model_version_id in self._model_version_metadata.keys()
                and model_version_id.model_id in self._model_metadata.keys()
            )

    @override
    def check_sub_model_existence(
        self,
        sub_model_id: SubModelId,
    ) -> bool:
        with self.lock:
            return sub_model_id in self.sub_model_metadata.keys()
