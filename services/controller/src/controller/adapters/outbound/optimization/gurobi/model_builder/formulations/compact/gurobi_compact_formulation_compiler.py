from typing import override

import gurobipy as gp

from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.collaborators import (
    CompactAssignmentBuilder,
    CompactComputeQueueBuilder,
    CompactExecutionBuilder,
    CompactMemoryBuilder,
    CompactNetworkBuilder,
    CompactObjectiveBuilder,
    CompactResponseTimeBuilder,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.compact_variables import (
    CompactVariables,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.gurobi_formulation_compiler import (
    GurobiFormulationCompiler,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)


class GurobiCompactFormulationCompiler(
    GurobiFormulationCompiler[CompactFormulationData, CompactVariables]
):
    """Create block variables, then connect them through explicit constraint inputs."""

    def __init__(
        self,
        *,
        assignment_builder: CompactAssignmentBuilder | None = None,
        execution_builder: CompactExecutionBuilder | None = None,
        compute_queue_builder: CompactComputeQueueBuilder | None = None,
        network_builder: CompactNetworkBuilder | None = None,
        memory_builder: CompactMemoryBuilder | None = None,
        response_time_builder: CompactResponseTimeBuilder | None = None,
        objective_builder: CompactObjectiveBuilder | None = None,
    ) -> None:
        self._assignment = (
            assignment_builder
            if assignment_builder is not None
            else CompactAssignmentBuilder()
        )
        self._execution = (
            execution_builder
            if execution_builder is not None
            else CompactExecutionBuilder()
        )
        self._compute_queue = (
            compute_queue_builder
            if compute_queue_builder is not None
            else CompactComputeQueueBuilder()
        )
        self._network = (
            network_builder if network_builder is not None else CompactNetworkBuilder()
        )
        self._memory = (
            memory_builder if memory_builder is not None else CompactMemoryBuilder()
        )
        self._response_time = (
            response_time_builder
            if response_time_builder is not None
            else CompactResponseTimeBuilder()
        )
        self._objective = (
            objective_builder
            if objective_builder is not None
            else CompactObjectiveBuilder()
        )

    def _create_variables(
        self, model: gp.Model, data: CompactFormulationData
    ) -> CompactVariables:
        return CompactVariables(
            assignment=self._assignment.create_variables(model, data),
            execution=self._execution.create_variables(model, data),
            compute_queue=self._compute_queue.create_variables(model, data),
            network=self._network.create_variables(model, data),
            memory=self._memory.create_variables(model, data),
            qos=self._response_time.create_variables(model, data),
        )

    @override
    def compile_formulation(
        self, model: gp.Model, data: CompactFormulationData
    ) -> CompactVariables:
        variables = self._create_variables(model, data)
        self._assignment.add_constraints(model, data, assignment=variables.assignment)
        self._execution.add_constraints(
            model, data, assignment=variables.assignment, execution=variables.execution
        )
        self._compute_queue.add_constraints(
            model,
            data,
            assignment=variables.assignment,
            execution=variables.execution,
            compute_queue=variables.compute_queue,
        )
        self._network.add_constraints(
            model, data, assignment=variables.assignment, network=variables.network
        )
        self._memory.add_constraints(
            model, data, assignment=variables.assignment, memory=variables.memory
        )
        self._response_time.add_constraints(
            model,
            data,
            assignment=variables.assignment,
            execution=variables.execution,
            compute_queue=variables.compute_queue,
            network=variables.network,
            qos=variables.qos,
        )
        self._objective.set_objective(
            model, data, assignment=variables.assignment, qos=variables.qos
        )
        return variables
