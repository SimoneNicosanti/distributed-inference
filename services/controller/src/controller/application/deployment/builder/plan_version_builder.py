from shared.plan.plan_version import PlanVersion


class PlanVersionBuilder:
    def __init__(self) -> None:
        self._current_plan_version = 0

    def build_next_plan_version(self) -> PlanVersion:
        self._current_plan_version += 1
        return PlanVersion(version_number=self._current_plan_version)
