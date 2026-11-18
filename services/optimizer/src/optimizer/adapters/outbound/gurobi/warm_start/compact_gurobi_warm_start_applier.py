from typing import override

from gurobipy import GRB

from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.warm_start.compact_solution_decisions import (
    compact_solution_decisions,
)
from optimizer.application.strategies.alternated.abc.warm_start_applier import (
    WarmStartApplier,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)


class CompactGurobiWarmStartApplier(
    WarmStartApplier[
        GurobiCompiledFormulation[MisocpCompactFormulation, CompactVariables]
    ]
):
    """Transfer decisions; target-only flows and all auxiliaries stay unspecified."""

    @override
    async def apply_warm_start(
        self,
        formulation_model: GurobiCompiledFormulation[
            MisocpCompactFormulation, CompactVariables
        ],
        warm_start_solution: OptimizationSolution,
    ) -> None:
        self._apply(formulation_model, warm_start_solution, allow_partial=False)

    async def apply_partial_warm_start(
        self,
        formulation_model: GurobiCompiledFormulation[
            MisocpCompactFormulation, CompactVariables
        ],
        warm_start_solution: OptimizationSolution,
    ) -> None:
        self._apply(formulation_model, warm_start_solution, allow_partial=True)

    @staticmethod
    def _apply(
        model: GurobiCompiledFormulation[MisocpCompactFormulation, CompactVariables],
        solution: OptimizationSolution,
        *,
        allow_partial: bool,
    ) -> None:
        decisions = compact_solution_decisions(
            model, solution, allow_partial=allow_partial
        )
        native = model.native_model
        variables = native.getVars()
        native.setAttr(GRB.Attr.Start, variables, [GRB.UNDEFINED] * len(variables))
        native.setAttr(
            GRB.Attr.Start, [v for v, _ in decisions], [value for _, value in decisions]
        )
        native.update()
