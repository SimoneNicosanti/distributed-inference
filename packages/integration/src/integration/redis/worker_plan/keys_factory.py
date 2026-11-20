from abc import ABC, abstractmethod
from typing import override

from shared.plan.plan_version import PlanVersion
from shared.service.service import WorkerId


class WorkerPlanKeyFactory(ABC):
    @staticmethod
    @abstractmethod
    def build_hash_key(version: PlanVersion) -> str: ...

    @staticmethod
    @abstractmethod
    def build_field_key(worker_id: WorkerId) -> str: ...


class StringWorkerPlanKeyFactory(WorkerPlanKeyFactory):
    @staticmethod
    @override
    def build_hash_key(version: PlanVersion) -> str:
        return f"worker-plans:version-{version.version_number}"

    @staticmethod
    @override
    def build_field_key(worker_id: WorkerId) -> str:
        return f"{worker_id.id}"
