from abc import ABC, abstractmethod
from typing import override

from shared.flow.flow import FlowId
from shared.plan.plan_version import PlanVersion


class FlowPlanKeyFactory(ABC):
    @staticmethod
    @abstractmethod
    async def build_flow_plans_dict_key(flow_id: FlowId) -> str: ...

    @staticmethod
    @abstractmethod
    async def build_flow_plan_elem_key(plan_version: PlanVersion) -> str: ...

    @staticmethod
    @abstractmethod
    async def build_current_plan_key(flow_id: FlowId) -> str: ...


class StringFlowPlanKeyFactory(FlowPlanKeyFactory):
    @staticmethod
    @override
    async def build_flow_plans_dict_key(flow_id: FlowId) -> str:
        return f"flow-plans:{flow_id.flow_id.hex}:versions"

    @staticmethod
    @override
    async def build_flow_plan_elem_key(plan_version: PlanVersion) -> str:
        return f"plan-version-{plan_version.version_number}"

    @staticmethod
    @override
    async def build_current_plan_key(flow_id: FlowId) -> str:
        return f"flow-plans:{flow_id.flow_id.hex}:active-version"
