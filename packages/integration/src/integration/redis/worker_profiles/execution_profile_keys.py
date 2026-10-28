import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import override

from integration.shared.id_json import model_variant_id_to_json
from shared.model.model_variant import ModelVariantId
from shared.service.service import WorkerId


class ExecutionProfileKeyFactory(ABC):
    @staticmethod
    @abstractmethod
    async def build_execution_profile_key_per_worker(worker_id: WorkerId) -> str: ...

    @staticmethod
    @abstractmethod
    async def build_execution_profile_ids_key_per_worker(
        worker_id: WorkerId,
    ) -> str: ...

    @staticmethod
    @abstractmethod
    async def build_model_execution_profile_key(
        model_variant_id: ModelVariantId,
    ) -> str: ...

    @staticmethod
    async def build_whole_execution_profile_key() -> str:
        return "worker:*:model-execution-profiles"


class HashExecutionProfileKeyFactory(ExecutionProfileKeyFactory):
    @staticmethod
    async def _build_worker_id_hash(worker_id: WorkerId) -> str:
        md5 = await asyncio.to_thread(hashlib.md5, str(worker_id.id).encode("utf-8"))

        return md5.hexdigest()

    @staticmethod
    @override
    async def build_execution_profile_key_per_worker(worker_id: WorkerId) -> str:
        worker_id_hash = await HashExecutionProfileKeyFactory._build_worker_id_hash(
            worker_id
        )
        return f"worker:{worker_id_hash}:model-execution-profiles"

    @staticmethod
    @override
    async def build_execution_profile_ids_key_per_worker(worker_id: WorkerId) -> str:
        worker_id_hash = await HashExecutionProfileKeyFactory._build_worker_id_hash(
            worker_id
        )
        return f"worker:{worker_id_hash}:model-execution-profile-ids"

    @staticmethod
    @override
    async def build_model_execution_profile_key(
        model_variant_id: ModelVariantId,
    ) -> str:
        model_variant_id_md5 = await asyncio.to_thread(
            hashlib.md5,
            model_variant_id_to_json(model_variant_id).encode("utf-8"),
        )
        return model_variant_id_md5.hexdigest()


class StringExecutionProfileKeyFactory(ExecutionProfileKeyFactory):
    @staticmethod
    @override
    async def build_execution_profile_key_per_worker(worker_id: WorkerId) -> str:
        return f"worker:{worker_id.id}:model-execution-profiles"

    @staticmethod
    @override
    async def build_execution_profile_ids_key_per_worker(worker_id: WorkerId) -> str:
        return f"worker:{worker_id.id}:model-execution-profile-ids"

    @staticmethod
    @override
    async def build_model_execution_profile_key(
        model_variant_id: ModelVariantId,
    ) -> str:
        return model_variant_id_to_json(model_variant_id)
