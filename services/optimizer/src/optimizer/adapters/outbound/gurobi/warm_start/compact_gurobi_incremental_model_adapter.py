import math
from typing import override

from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.warm_start.compact_gurobi_warm_start_applier import (
    CompactGurobiWarmStartApplier,
)
from optimizer.application.strategies.incremental.incremental_model_adapter import (
    IncrementalModelAdapter,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.solution.optimization_solution import OptimizationSolution

type CompactGurobiModel = GurobiCompiledFormulation[
    MisocpCompactFormulation, CompactVariables
]


class CompactGurobiIncrementalModelAdapter(
    IncrementalModelAdapter[CompactGurobiModel, CompactGurobiModel]
):
    def __init__(self) -> None:
        self._warm_start_applier = CompactGurobiWarmStartApplier()

    @override
    async def initialize_model(
        self,
        model: CompactGurobiModel,
        solution: OptimizationSolution,
    ) -> None:
        await self._warm_start_applier.apply_partial_warm_start(model, solution)

    @override
    def is_valid_solution(self, model: CompactGurobiModel) -> bool:
        native = model.native_model
        if native.SolCount <= 0:
            return False
        return all(
            math.isfinite(value) and value <= tolerance
            for value, tolerance in (
                (float(native.ConstrVio), float(native.Params.FeasibilityTol)),
                (float(native.BoundVio), float(native.Params.FeasibilityTol)),
                (float(native.IntVio), float(native.Params.IntFeasTol)),
            )
        )
