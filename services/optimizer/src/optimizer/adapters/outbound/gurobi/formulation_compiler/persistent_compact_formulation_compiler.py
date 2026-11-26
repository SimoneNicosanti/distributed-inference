from collections.abc import AsyncGenerator
from contextlib import ExitStack, asynccontextmanager
from typing import override

import gurobipy as gp

from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.persistent_compact_model import (
    PersistentCompactModel,
)
from optimizer.application.ports.outbound.persistent_formulation_compiler import (
    PersistentFormulationCompiler,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)


class GurobiPersistentCompactFormulationCompiler(
    PersistentFormulationCompiler[
        MisocpCompactFormulation,
        GurobiCompiledFormulation[MisocpCompactFormulation, CompactVariables],
    ]
):
    def __init__(self, *, output_flag: bool = False) -> None:
        self._output_flag = output_flag
        self._session: PersistentCompactModel | None = None
        self._resources: ExitStack | None = None

    @override
    @asynccontextmanager
    async def open_session(self) -> AsyncGenerator[None]:
        if self._resources is not None:
            raise RuntimeError("A persistent compilation session is already active")
        with ExitStack() as resources:
            self._resources = resources
            try:
                yield None
            finally:
                self._session = None
                self._resources = None

    @override
    @asynccontextmanager
    async def compile_formulation(
        self, formulation: MisocpCompactFormulation
    ) -> AsyncGenerator[
        GurobiCompiledFormulation[MisocpCompactFormulation, CompactVariables]
    ]:
        resources = self._resources
        if resources is None:
            raise RuntimeError("Persistent compilation requires an active session")
        session = self._session
        if session is None:
            # Include environment/model setup in the first prefix's construction
            # measurement, exactly as the fresh-model compiler does.
            environment = resources.enter_context(gp.Env(empty=True))
            environment.setParam("OutputFlag", int(self._output_flag))
            environment.start()
            model = resources.enter_context(
                gp.Model("distributed_inference_persistent", env=environment)
            )
            model.Params.NonConvex = 0
            session = self._session = PersistentCompactModel(model)
        yield session.extend(formulation)
