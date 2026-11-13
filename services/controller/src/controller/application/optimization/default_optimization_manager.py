from typing import override

from controller.application.optimization.abc.optimization_context_builder import (
    OptimizationContextBuilder,
)
from controller.application.optimization.abc.optimization_problem_factory import (
    OptimizationProblemFactory,
)
from controller.application.optimization.abc.optimization_solution_enricher import (
    OptimizationSolutionEnricher,
)
from controller.application.ports.inbound.optimization.optimization_manager import (
    OptimizationManager,
)
from controller.application.ports.outbound.optimization.optimizer_gateway import (
    OptimizerGateway,
)
from controller.application.ports.outbound.store.optimization_solution_store import (
    OptimizationSolutionStore,
)
from controller.domain.optimization.request.optimization_request import (
    OptimizationRequestsByFlow,
)


class DefaultOptimizationManager(OptimizationManager):
    def __init__(
        self,
        optimization_context_builder: OptimizationContextBuilder,
        optimization_problem_factory: OptimizationProblemFactory,
        optimizer_gateway: OptimizerGateway,
        solution_enricher: OptimizationSolutionEnricher,
        solution_store: OptimizationSolutionStore,
    ) -> None:
        self._optimization_context_builder = optimization_context_builder
        self._optimization_problem_factory = optimization_problem_factory
        self._optimizer_gateway = optimizer_gateway
        self._solution_enricher = solution_enricher
        self._solution_store = solution_store

    @override
    async def optimize_inference_distribution(
        self,
        requests_by_flow: OptimizationRequestsByFlow,
    ) -> None:
        context = await self._optimization_context_builder.build_optimization_context()
        problem = await self._optimization_problem_factory.build_optimization_problem(
            requests_by_flow, context
        )
        solution = await self._optimizer_gateway.optimize(problem)
        enriched_solution = self._solution_enricher.enrich_solution(
            solution,
            context.static_profiles_by_variant,
        )
        await self._solution_store.put_solution(enriched_solution)
