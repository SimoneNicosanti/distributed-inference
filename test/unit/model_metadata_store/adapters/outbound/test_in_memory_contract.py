from typing import override

import pytest

from distributed_inference.model_metadata_store.adapters.outbound.in_memory_model_metadata_store import (
    InMemoryModelMetadataStore,
)
from distributed_inference.model_metadata_store.application.ports.outbound.model_metadata_store import (
    ModelMetadataStore,
)
from test.contracts.model_metadata_store.model_metadata_store_contract import (
    ModelMetadataStoreContract,
)

pytestmark = pytest.mark.contract


class TestInMemoryModelMetadataStoreContract(ModelMetadataStoreContract):
    @override
    def build_store(self) -> ModelMetadataStore:
        return InMemoryModelMetadataStore()
