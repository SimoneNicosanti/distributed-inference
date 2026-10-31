from abc import ABC, abstractmethod
from typing import override

from shared.plan.plan_version import PlanVersion


class HopDeploymentKeyFactory(ABC):
    @staticmethod
    @abstractmethod
    async def build_hop_deployments_key(plan_version: PlanVersion) -> str: ...


class StringHopDeploymentKeyFactory(HopDeploymentKeyFactory):
    @staticmethod
    @override
    async def build_hop_deployments_key(plan_version: PlanVersion) -> str:
        return f"hop-deployments:{plan_version.version_number}"
