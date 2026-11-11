"""Incumbent observation contracts; no native model or license required."""

from unittest.mock import Mock

import pytest
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi.trackers.gurobi_incumbent_tracker import (
    GurobiIncumbentTracker,
)


def observe(tracker: GurobiIncumbentTracker, runtime_s: float, value: float) -> None:
    model = Mock()
    model.cbGet.side_effect = [runtime_s, value]
    tracker(model, GRB.Callback.MIPSOL)


@pytest.mark.parametrize(
    "minimize,values,expected",
    [
        (True, [10, 12, 10, 8, 6, 7], [None, 8, 8, 6, 6]),
        (False, [10, 8, 10, 12, 14, 13], [None, 12, 12, 14, 14]),
    ],
)
def test_trace_carries_forward_improvements_without_future_values(
    minimize: bool, values: list[float], expected: list[float | None]
) -> None:
    tracker = GurobiIncumbentTracker(minimize=minimize)
    for timestamp, value in zip([0.2, 0.3, 0.4, 0.8, 2.4, 2.8], values, strict=True):
        observe(tracker, timestamp, value)
    samples = tracker.samples(3.2, values[4])
    assert tracker.time_to_first_incumbent_s == 0.2
    assert [sample.runtime_s for sample in samples] == [0, 1, 2, 3, 3.2]
    assert [sample.objective_value for sample in samples] == expected


def test_exact_second_and_final_fraction_preserve_zero_incumbent() -> None:
    tracker = GurobiIncumbentTracker(minimize=True)
    observe(tracker, 1.0, 1.0)
    observe(tracker, 1.2, 0.0)
    samples = tracker.samples(1.5, 0.0)
    assert tracker.time_to_first_incumbent_s == 1.0
    assert [(sample.runtime_s, sample.objective_value) for sample in samples] == [
        (0.0, None),
        (1.0, 1.0),
        (1.5, 0.0),
    ]


def test_no_incumbent_has_null_values_and_no_first_discovery_time() -> None:
    tracker = GurobiIncumbentTracker(minimize=True)
    samples = tracker.samples(2.0, None)
    assert tracker.time_to_first_incumbent_s is None
    assert [sample.runtime_s for sample in samples] == [0, 1, 2]
    assert all(sample.objective_value is None for sample in samples)
    assert tracker.samples(None, None) == ()


def test_final_only_solution_does_not_invent_discovery_time() -> None:
    tracker = GurobiIncumbentTracker(minimize=True)
    samples = tracker.samples(0.4, 5.0)
    assert tracker.time_to_first_incumbent_s is None
    assert [(sample.runtime_s, sample.objective_value) for sample in samples] == [
        (0.0, None),
        (0.4, 5.0),
    ]


def test_non_solution_callbacks_and_invalid_candidates_are_ignored() -> None:
    tracker = GurobiIncumbentTracker(minimize=True)
    model = Mock()
    tracker(model, GRB.Callback.MIP)
    model.cbGet.assert_not_called()
    for value in (float("nan"), float("inf"), GRB.INFINITY, -GRB.INFINITY):
        observe(tracker, 0.1, value)
    observe(tracker, 0.2, 3.0)
    assert tracker.time_to_first_incumbent_s == 0.2
    assert tracker.samples(0.5, 3.0)[-1].objective_value == 3.0
