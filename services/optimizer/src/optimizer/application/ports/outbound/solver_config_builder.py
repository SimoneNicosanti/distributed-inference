from abc import ABC, abstractmethod


class SolverConfigBuilder[SolverConfigT](ABC):
    @abstractmethod
    def build_solver_config(
        self,
    ) -> SolverConfigT: ...
