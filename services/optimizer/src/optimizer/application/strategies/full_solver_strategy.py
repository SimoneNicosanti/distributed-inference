from typing import override

from optimizer.application.abc.formulation_builder import (
    FormulationBuilder,
)
from optimizer.application.abc.solver_strategy import (
    SolveError,
    SolverStrategy,
)
from optimizer.application.ports.outbound.formulation_compiler import (
    FormulationCompiler,
)
from optimizer.application.ports.outbound.solved_compiled_formulation_translator import (
    SolvedCompiledFormulationTranslator,
)
from optimizer.application.ports.outbound.solver import Solver
from optimizer.domain.formulation.formulation import Formulation
from optimizer.domain.formulation.formulation_config import FormulationConfig
from optimizer.domain.problem.optimization_problem import OptimizationProblem
from optimizer.domain.solution.optimization_solution import (
    OptimizationSolution,
)


class FullSolverStrategy[
    OptimizationProblemT: OptimizationProblem,
    FormulationT: Formulation,
    FormulationConfigT: FormulationConfig,
    CompiledFormulationT,
    SolverConfigT,
    SolvedCompiledFormulationT,
    SolverStatisticsT,
](SolverStrategy[OptimizationProblemT]):
    def __init__(
        self,
        formulation_builder: FormulationBuilder[
            OptimizationProblemT, FormulationT, FormulationConfigT
        ],
        formulation_config: FormulationConfigT,
        formulation_compiler: FormulationCompiler[FormulationT, CompiledFormulationT],
        solver_config: SolverConfigT,
        solver: Solver[
            CompiledFormulationT,
            SolverConfigT,
            SolvedCompiledFormulationT,
            SolverStatisticsT,
        ],
        solved_model_translator: SolvedCompiledFormulationTranslator[
            SolvedCompiledFormulationT
        ],
    ) -> None:
        super().__init__()
        self._formulation_builder = formulation_builder
        self._formulation_config = formulation_config

        self._formulation_compiler = formulation_compiler

        self._solver_config = solver_config
        self._solver = solver

        self._solved_model_translator = solved_model_translator

    @override
    async def solve(self, problem: OptimizationProblemT) -> OptimizationSolution:

        formulation = self._formulation_builder.build_formulation(
            problem, self._formulation_config
        )
        async with self._formulation_compiler.compile_formulation(
            formulation
        ) as formulation_model:
            solver_output = await self._solver.solve_compiled_formulation(
                formulation_model, self._solver_config
            )
            if solver_output.solved_model is not None:
                return await self._solved_model_translator.translate_solved_formulation_model(
                    solver_output.solved_model
                )

            raise SolveError()
