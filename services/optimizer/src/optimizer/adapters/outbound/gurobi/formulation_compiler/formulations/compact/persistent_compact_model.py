"""Append flow-local blocks while retaining previous native variables/constraints."""

from dataclasses import replace
from time import perf_counter

import gurobipy as gp

from optimizer.adapters.outbound.gurobi.entities.gurobi_compiled_formulation import (
    GurobiCompiledFormulation,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators import (
    CompactComputeQueueBuilder,
    CompactObjectiveBuilder,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.memory.constraints import (
    add_memory_capacity_constraints,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.collaborators.network.constraints import (
    add_network_queue_constraints,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.compact_variables import (
    CompactVariables,
)
from optimizer.adapters.outbound.gurobi.formulation_compiler.formulations.compact.gurobi_misocp_compact_formulation_compiler import (
    GurobiMisocpCompactFormulationModelBuilder,
)
from optimizer.domain.formulation.compact.misocp.misocp_compact_formulation import (
    MisocpCompactFormulation,
)
from shared.service.service import WorkerId


class PersistentCompactModel:
    def __init__(self, model: gp.Model) -> None:
        self.model = model
        self.data: MisocpCompactFormulation | None = None
        self.variables: CompactVariables | None = None
        self._compute = CompactComputeQueueBuilder()
        self._objective = CompactObjectiveBuilder()
        self._builder = GurobiMisocpCompactFormulationModelBuilder(
            compute_queue_builder=self._compute, objective_builder=self._objective
        )
        self._queue_linear: list[gp.Constr] = []
        self._queue_quadratic: list[gp.QConstr] = []
        self._network_rows: dict[WorkerId, gp.Constr] = {}
        self._memory_rows: dict[WorkerId, gp.Constr] = {}

    def extend(
        self, data: MisocpCompactFormulation
    ) -> GurobiCompiledFormulation[MisocpCompactFormulation, CompactVariables]:
        started = perf_counter()
        old = self.data
        if old is not None:
            self._validate_extension(old, data)
        new_flows = tuple(f for f in data.flows if old is None or f not in old.flows)
        indices = dict(old.flow_index) if old is not None else {}
        for flow in new_flows:
            indices[flow] = len(indices)
        data = replace(data, flow_index=indices)
        local = replace(data, flows=new_flows)
        existing = self.variables
        added = self._builder.create_variables(self.model, local, existing=existing)
        if existing is None:
            self.variables = added
        else:
            existing.extend(added)
        v = self.variables
        assert v is not None
        self._builder.add_constraints(self.model, local, v, include_shared=False)
        self.model.update()
        self._refresh_queue(data, v)
        if old is None:
            add_network_queue_constraints(
                self.model, data, network=v.network, include_local=False
            )
            add_memory_capacity_constraints(self.model, data, memory=v.memory)
            self.model.update()
            for worker in data.workers:
                network_row = self.model.getConstrByName(
                    f"network_utilization_value[{data.worker_index[worker]}]"
                )
                memory_row = self.model.getConstrByName(
                    f"memory_capacity[{data.worker_index[worker]}]"
                )
                assert network_row is not None and memory_row is not None
                self._network_rows[worker] = network_row
                self._memory_rows[worker] = memory_row
        else:
            for (
                flow,
                variant,
                _,
                worker,
            ), variable in added.network.transmission_time_s.items():
                self.model.chgCoeff(
                    self._network_rows[worker],
                    variable,
                    -data.service_arrival_rate_per_s[flow, variant],
                )
            for (*_, worker), variable in added.memory.component_weight_mb.items():
                self.model.chgCoeff(self._memory_rows[worker], variable, 1.0)
        self._objective.set_objective(
            self.model, data, assignment=v.assignment, qos=v.qos
        )
        self.model.update()
        self.data = data
        return GurobiCompiledFormulation(
            native_model=self.model,
            variables=v,
            formulation=data,
            construction_time_s=perf_counter() - started,
        )

    def _refresh_queue(
        self, data: MisocpCompactFormulation, v: CompactVariables
    ) -> None:
        # Only coupling/bound-dependent rows are replaced. Previous assignments,
        # execution rows, priority assignments and transmission rows stay native.
        self.model.remove(self._queue_linear)
        self.model.remove(self._queue_quadratic)
        self.model.update()
        for (flow, _), variable in v.compute_queue.component_queue_wait_s.items():
            variable.UB = data.max_queue_wait_s_by_flow[flow]
        for (worker, _), variable in v.compute_queue.weighted_queue_wait_s.items():
            variable.UB = data.max_queue_wait_s_by_worker[worker]
        linear_count, quadratic_count = self.model.NumConstrs, self.model.NumQConstrs
        self._compute.add_constraints(
            self.model,
            data,
            assignment=v.assignment,
            execution=v.execution,
            compute_queue=v.compute_queue,
            include_local=False,
        )
        self.model.update()
        self._queue_linear = self.model.getConstrs()[linear_count:]
        self._queue_quadratic = self.model.getQConstrs()[quadratic_count:]

    @staticmethod
    def _validate_extension(
        old: MisocpCompactFormulation, new: MisocpCompactFormulation
    ) -> None:
        if not set(old.flows) < set(new.flows):
            raise ValueError("Persistent compilation requires a strict flow extension")
        if any(old.problem.flows[f] != new.problem.flows[f] for f in old.flows):
            raise ValueError("Existing flow requests must remain unchanged")
        for name in (
            "representation",
            "workers",
            "compute_bin_limits",
            "network_bin_limits",
        ):
            if getattr(old, name) != getattr(new, name):
                raise ValueError(f"Persistent compilation cannot change {name}")
        for name in (
            "policy",
            "workers",
            "connections",
            "variants",
            "configurations",
            "execution_times_s",
        ):
            previous, current = getattr(old.problem, name), getattr(new.problem, name)
            if previous is not current and previous != current:
                raise ValueError(f"Persistent compilation cannot change {name}")
