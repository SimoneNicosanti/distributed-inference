from typing import override

import pytest

# Adatta soltanto questo import al percorso reale.
from distributed_inference.adapters.outbound.model_metadata_store.in_memory import (
    InMemoryModelMetadataStore,
)
from distributed_inference.application.model_metadata_store.contracts.model_metadata_store import (
    ModelMetadataStore,
)
from test.contracts.model_metadata_store_contract import (
    ModelMetadataStoreContract,
)

pytestmark = pytest.mark.contract


class TestInMemoryModelMetadataStoreContract(ModelMetadataStoreContract):
    @override
    def build_store(self) -> ModelMetadataStore:
        return InMemoryModelMetadataStore()
