"""Observe incumbent improvements and reconstruct a one-second step trace."""

import math

import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi.entities.gurobi_incumbent_sample import (
    GurobiIncumbentSample,
)


class GurobiIncumbentTracker:
    def __init__(self, *, minimize: bool) -> None:
        self._minimize = minimize
        self._improvements: list[GurobiIncumbentSample] = []
        self.time_to_first_incumbent_s: float | None = None

    def _record(self, runtime_s: float, value: float) -> bool:
        if (
            not math.isfinite(runtime_s)
            or runtime_s < 0
            or not math.isfinite(value)
            or abs(value) >= GRB.INFINITY
        ):
            return False
        if self._improvements:
            previous = self._improvements[-1].objective_value
            assert previous is not None
            if (self._minimize and value >= previous) or (
                not self._minimize and value <= previous
            ):
                return False
        self._improvements.append(GurobiIncumbentSample(runtime_s, value))
        return True

    def __call__(self, model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        runtime_s = float(model.cbGet(GRB.Callback.RUNTIME))
        value = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
        if self._record(runtime_s, value) and self.time_to_first_incumbent_s is None:
            self.time_to_first_incumbent_s = runtime_s

    def samples(
        self, runtime_s: float | None, final_value: float | None
    ) -> tuple[GurobiIncumbentSample, ...]:
        """Carry forward only values observed at or before each sample time.

        A final native incumbent missed by callbacks is visible only at termination;
        its discovery time cannot be inferred, so it never sets time-to-first.
        """
        if runtime_s is None:
            return ()
        if final_value is not None:
            self._record(runtime_s, final_value)
        times = [float(second) for second in range(math.floor(runtime_s) + 1)]
        if not times or times[-1] != runtime_s:
            times.append(runtime_s)
        samples = []
        index = 0
        value = None
        for timestamp in times:
            while (
                index < len(self._improvements)
                and self._improvements[index].runtime_s <= timestamp
            ):
                value = self._improvements[index].objective_value
                index += 1
            samples.append(GurobiIncumbentSample(timestamp, value))
        return tuple(samples)
