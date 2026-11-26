"""Views of existing generators; preserve total nominal load at every F."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from typing import Any

from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)
from scalability_benchmark.bottleneck.configuration import Case
from scalability_benchmark.flows import flow_pool
from scalability_benchmark.instances import Scenario, make_instance
from scalability_benchmark.randomness import fingerprint


def jsonable(value: Any) -> Any:
    """Canonical records including domain mappings with structured keys."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: jsonable(item) for key, item in sorted(value.items())}
        return sorted(
            [[jsonable(key), jsonable(item)] for key, item in value.items()],
            key=lambda pair: fingerprint(pair[0]),
        )
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted([jsonable(item) for item in value], key=fingerprint)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def instance(scenario: Scenario, case: Case) -> CompactOptimizationProblem:
    config = scenario.config.model_copy(
        update={
            "flows": scenario.config.flows.model_copy(
                update={"pool_size": case.spec.flow_count}
            ),
            "optimization": scenario.config.optimization.model_copy(
                update={
                    "max_components": case.max_components,
                    "priority_count": case.priority_count,
                }
            ),
        }
    )
    flows = flow_pool(
        config,
        case.spec.seed,
        case.spec.load,
        scenario.reference_time_s,
        scenario.workers[0].id,
    )
    view = replace(scenario, config=config, flows={case.spec.load: flows})
    problem = make_instance(view, case.spec)
    if len(problem.workers) != case.spec.worker_count:
        raise ValueError("Requested infrastructure is not present in the worker pool")
    return problem
