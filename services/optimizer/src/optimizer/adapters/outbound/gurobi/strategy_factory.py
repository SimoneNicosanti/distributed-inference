from dataclasses import replace

from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_run_statistics import (
    GurobiRunStatistics,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_solver_config import (
    GurobiSolverConfig,
)
from optimizer.adapters.outbound.gurobi.entities.gurobi_strategy_config import (
    GurobiStrategyConfig,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.gurobi_misocp_compact_formulation_compiler import (
    GurobiMisocpCompactFormulationModelBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.gurobi_formulation_compiler import (
    GurobiFormulationCompiler,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.persistent_compact_formulation_compiler import (
    GurobiPersistentCompactFormulationCompiler,
)
from optimizer.adapters.outbound.gurobi.result_translation.formulations.compact.gurobi_misocp_compact_solved_compiled_formulation_translator import (
    GurobiMisocpCompactSolvedCompiledFormulationTranslator,
)
from optimizer.adapters.outbound.gurobi.solver.gurobi_solver import GurobiSolver
from optimizer.adapters.outbound.gurobi.warm_start.compact_gurobi_incremental_model_adapter import (
    CompactGurobiIncrementalModelAdapter,
)
from optimizer.application.abc.solver_strategy import SolverStrategy
from optimizer.application.formulation.compact.misocp.misocp_compact_formulation_builder import (
    MisocpCompactFormulationBuilder,
)
from optimizer.application.ports.outbound.solver import Solver
from optimizer.application.strategies.full_solver_strategy import FullSolverStrategy
from optimizer.application.strategies.incremental.incremental_flow_solver_strategy import (
    IncrementalFlowSolverStrategy,
    IncrementalFlowStrategyConfig,
)
from optimizer.application.strategies.incremental.persistent_incremental_flow_solver_strategy import (
    PersistentIncrementalFlowSolverStrategy,
)
from optimizer.application.strategies.solver_strategy_kind import SolverStrategyKind
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation_config import (
    MisocpCompactFormulationConfig,
)
from optimizer.domain.problem.compact.compact_optimization_problem import (
    CompactOptimizationProblem,
)

type CompactGurobiModel = GurobiCompiledFormulation[
    MisocpCompactFormulation, CompactVariables
]
type CompactGurobiSolver = Solver[
    CompactGurobiModel, GurobiSolverConfig, CompactGurobiModel, GurobiRunStatistics
]


def build_compact_gurobi_solver_strategy(
    formulation_config: MisocpCompactFormulationConfig,
    solver_config: GurobiSolverConfig,
    *,
    strategy_config: GurobiStrategyConfig | None = None,
    solver: CompactGurobiSolver | None = None,
) -> SolverStrategy[CompactOptimizationProblem]:
    """Optimizer-owned composition; no controller or transport configuration required."""
    strategy_config = strategy_config or GurobiStrategyConfig()
    builder = MisocpCompactFormulationBuilder()
    compiler = GurobiFormulationCompiler(
        GurobiMisocpCompactFormulationModelBuilder(),
        non_convex=0,
        output_flag=solver_config.output_flag,
    )
    native_solver = (
        solver
        if solver is not None
        else GurobiSolver[MisocpCompactFormulation, CompactVariables]()
    )
    translator = GurobiMisocpCompactSolvedCompiledFormulationTranslator()
    if strategy_config.strategy == SolverStrategyKind.FULL:
        return FullSolverStrategy(
            formulation_builder=builder,
            formulation_config=formulation_config,
            formulation_compiler=compiler,
            solver_config=solver_config,
            solver=native_solver,
            solved_model_translator=translator,
        )
    persistent_compiler = (
        GurobiPersistentCompactFormulationCompiler(
            output_flag=solver_config.output_flag
        )
        if strategy_config.strategy == SolverStrategyKind.INCREMENTAL_PERSISTENT
        else None
    )
    incremental = IncrementalFlowSolverStrategy(
        formulation_builder=builder,
        formulation_config=formulation_config,
        formulation_compiler=persistent_compiler
        if persistent_compiler is not None
        else compiler,
        solver=native_solver,
        solved_model_translator=translator,
        model_adapter=CompactGurobiIncrementalModelAdapter(),
        config=IncrementalFlowStrategyConfig(
            insertion_solver_config=replace(
                solver_config,
                time_limit_s=strategy_config.insertion_time_limit_s
                or solver_config.time_limit_s,
                solution_limit=None,
                record_incumbent_history=True,
            ),
        ),
    )
    if persistent_compiler is not None:
        return PersistentIncrementalFlowSolverStrategy(incremental, persistent_compiler)
    return incremental
