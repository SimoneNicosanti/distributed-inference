from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import override

import gurobipy as gp

from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.abc.gurobi_model_builder import (
    GurobiModelBuilder,
)
from optimizer.application.ports.outbound.formulation_compiler import (
    FormulationCompiler,
)
from optimizer.domain.formulation.formulation import Formulation


class GurobiFormulationCompiler[FormulationT: Formulation, FormulationVariablesT](
    FormulationCompiler[
        FormulationT,
        GurobiCompiledFormulation[FormulationT, FormulationVariablesT],
    ]
):
    """Own native resources and delegate mathematical construction to a formulation."""

    def __init__(
        self,
        gurobi_model_builder: GurobiModelBuilder[FormulationT, FormulationVariablesT],
        *,
        non_convex: int,
        output_flag: bool = False,
    ) -> None:
        self._non_convex = non_convex
        self._gurobi_model_builder = gurobi_model_builder
        self._output_flag = output_flag

    @override
    @asynccontextmanager
    async def compile_formulation(
        self, formulation: FormulationT
    ) -> AsyncGenerator[GurobiCompiledFormulation[FormulationT, FormulationVariablesT]]:
        construction_started = perf_counter()
        with gp.Env(empty=True) as environment:
            environment.setParam("OutputFlag", int(self._output_flag))
            environment.start()
            with gp.Model("distributed_inference", env=environment) as native_model:
                native_model.Params.NonConvex = self._non_convex
                variables = self._gurobi_model_builder.build_model_from_formulation(
                    native_model, formulation
                )
                native_model.update()
                yield GurobiCompiledFormulation(
                    native_model=native_model,
                    variables=variables,
                    formulation=formulation,
                    construction_time_s=perf_counter() - construction_started,
                )
