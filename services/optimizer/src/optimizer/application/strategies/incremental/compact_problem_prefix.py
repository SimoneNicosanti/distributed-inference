from copy import copy
from itertools import islice
from types import MappingProxyType

from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)


def compact_problem_prefix(
    problem: CompactOptimizationProblem, flow_count: int
) -> CompactOptimizationProblem:
    """Share validated immutable profiles and policies; only project the flow map."""
    if type(flow_count) is not int or not 1 <= flow_count <= len(problem.flows):
        raise ValueError(
            "A flow prefix must be nonempty and within the original problem"
        )
    if flow_count == len(problem.flows):
        return problem
    prefix = copy(problem)
    flows = dict(islice(problem.flows.items(), flow_count))
    # dataclasses.replace would deep-copy execution profiles in __post_init__.
    object.__setattr__(prefix, "flows", MappingProxyType(flows))
    return prefix
