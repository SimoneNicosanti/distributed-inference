import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import override

from shared.service.service import WorkerId


class NetworkProfileKeyFactory(ABC):
    @staticmethod
    @abstractmethod
    async def build_network_profile_key_per_worker(worker_id: WorkerId) -> str: ...

    @staticmethod
    @abstractmethod
    async def build_whole_network_profile_key() -> str: ...


class HashNetworkProfileKeyFactory(NetworkProfileKeyFactory):
    @staticmethod
    @override
    async def build_network_profile_key_per_worker(worker_id: WorkerId) -> str:
        worker_id_hash = await asyncio.to_thread(
            hashlib.md5, str(worker_id.id).encode("utf-8")
        )
        worker_id_hash_dig = worker_id_hash.hexdigest()
        return f"worker:{worker_id_hash_dig}:network-profile"

    @staticmethod
    @override
    async def build_whole_network_profile_key() -> str:
        return "worker:*:network-profile"


class StringNetworkProfileKeyFactory(NetworkProfileKeyFactory):
    @staticmethod
    @override
    async def build_network_profile_key_per_worker(worker_id: WorkerId) -> str:
        return f"worker:{worker_id.id}:network-profile"

    @staticmethod
    @override
    async def build_whole_network_profile_key() -> str:
        return "worker:*:network-profile"
