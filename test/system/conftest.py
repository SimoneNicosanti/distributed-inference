import os
from pathlib import Path

import pytest
from test.system.existing_plan_scenario import (
    ExistingPlanScenario,
    load_existing_plan_scenario,
)
from test.system.optimization_readiness import wait_for_http
from test.system.optimization_scenario import (
    OptimizationScenario,
    prepare_optimization_scenario,
)

from integration.front_end.http.front_end_http_paths import (
    FRONT_END_LOCAL_WORKER_IDENTITY_PATH,
)


@pytest.fixture(scope="session")
def optimization_scenario(
    tmp_path_factory: pytest.TempPathFactory,
) -> OptimizationScenario:
    """Create a new scenario for the optimization request test."""
    for variable in ("MODEL_MANAGER_BASE_URL", "CONTROLLER_BASE_URL"):
        if url := os.getenv(variable):
            wait_for_http(url, "/health/ready")
    workspace: Path = tmp_path_factory.mktemp("optimization-scenario")
    return prepare_optimization_scenario(workspace)


@pytest.fixture(scope="session")
def existing_plan_scenario() -> ExistingPlanScenario:
    return load_existing_plan_scenario()


@pytest.fixture(scope="session", autouse=True)
def system_stack_ready() -> None:
    base_url = os.getenv("FRONT_END_BASE_URL")
    if base_url is None:
        return
    wait_for_http(base_url, FRONT_END_LOCAL_WORKER_IDENTITY_PATH)
