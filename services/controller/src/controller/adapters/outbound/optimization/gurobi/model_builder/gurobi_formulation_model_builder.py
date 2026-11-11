from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import override

import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi.entities.gurobi_formulation_model import (
    GurobiFormulationModel,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.gurobi_formulation_compiler import (
    GurobiFormulationCompiler,
)
from controller.application.ports.outbound.optimization.formulation_model_builder import (
    FormulationModelBuilder,
)


class GurobiFormulationModelBuilder[FormulationDataT, FormulationVariablesT](
    FormulationModelBuilder[
        FormulationDataT,
        GurobiFormulationModel[FormulationDataT, FormulationVariablesT],
    ]
):
    """Own native resources and delegate mathematical construction to a formulation."""

    def __init__(
        self,
        formulation_compiler: GurobiFormulationCompiler[
            FormulationDataT, FormulationVariablesT
        ],
        *,
        non_convex: int,
        output_flag: bool = False,
    ) -> None:
        self._non_convex = non_convex
        self._formulation_compiler = formulation_compiler
        self._output_flag = output_flag

    @override
    @asynccontextmanager
    async def build_formulation_model(
        self, data: FormulationDataT
    ) -> AsyncGenerator[
        GurobiFormulationModel[FormulationDataT, FormulationVariablesT]
    ]:
        construction_started = perf_counter()
        with gp.Env(empty=True) as environment:
            environment.setParam("OutputFlag", int(self._output_flag))
            environment.start()
            with gp.Model("distributed_inference", env=environment) as native_model:
                native_model.Params.NonConvex = self._non_convex
                variables = self._formulation_compiler.compile_formulation(
                    native_model, data
                )
                native_model.update()
                yield GurobiFormulationModel(
                    native_model=native_model,
                    variables=variables,
                    data=data,
                    construction_time_s=perf_counter() - construction_started,
                )
