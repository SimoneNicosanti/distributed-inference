from collections.abc import Callable
from typing import override

from optimizer.adapters.outbound.gurobi.entities.gurobi_run_statistics import (
    GurobiRunStatistics,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.application.ports.outbound.solver_config_builder import (
    SolverConfigBuilder,
)


class GurobiSolverConfigBuilder(SolverConfigBuilder[GurobiSolverConfig]):
    def __init__(
        self,
        *,
        output_flag: bool = False,
        time_limit_s: float | None = None,
        mip_gap: float | None = None,
        threads: int | None = None,
        seed: int | None = None,
        observer: Callable[[GurobiRunStatistics], None] | None = None,
        record_incumbent_history: bool = False,
    ) -> None:
        super().__init__()
        self._output_flag = output_flag
        self._time_limit_s = time_limit_s
        self._mip_gap = mip_gap
        self._threads = threads
        self._seed = seed
        self._observer = observer
        self._record_incumbent_history = record_incumbent_history

    @override
    def build_solver_config(self) -> GurobiSolverConfig:
        return GurobiSolverConfig(
            output_flag=self._output_flag,
            time_limit_s=self._time_limit_s,
            mip_gap=self._mip_gap,
            threads=self._threads,
            seed=self._seed,
            observer=self._observer,
            record_incumbent_history=self._record_incumbent_history,
        )
