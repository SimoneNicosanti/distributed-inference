from typing import override

import gurobipy as gp
from gurobipy import GRB

from controller.adapters.outbound.optimization.gurobi.entities.gurobi_formulation_model import (
    GurobiFormulationModel,
)
from controller.adapters.outbound.optimization.gurobi.model_builder.formulations.compact.compact_variables import (
    CompactVariables,
)
from controller.application.optimization.strategies.alternated.abc.warm_start_applier import (
    WarmStartApplier,
)
from controller.domain.optimization.solving.formulation.compact.compact_formulation_data import (
    CompactFormulationData,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)


class CompactGurobiWarmStartApplier(
    WarmStartApplier[GurobiFormulationModel[CompactFormulationData, CompactVariables]]
):
    """Transfer incumbent decisions, leaving auxiliary starts undefined."""

    @override
    async def apply_warm_start(
        self,
        formulation_model: GurobiFormulationModel[
            CompactFormulationData, CompactVariables
        ],
        warm_start_solution: OptimizationSolution,
    ) -> None:
        data = formulation_model.data
        assignment = formulation_model.variables.assignment
        compute_queue = formulation_model.variables.compute_queue
        flow_solutions = warm_start_solution.flow_solutions
        if len(flow_solutions) != len(data.flows) or {
            solution.flow_id for solution in flow_solutions
        } != set(data.flows):
            raise ValueError("Warm start must contain exactly the target model flows")

        decision_variables = [
            variable
            for group in (
                assignment.model_selected,
                assignment.layer_component,
                assignment.component_worker,
                assignment.component_active,
                assignment.component_last,
                compute_queue.component_priority,
            )
            for variable in group.values()
        ]
        selected_variables: list[gp.Var] = []
        for solution in flow_solutions:
            flow = solution.flow_id
            variant = solution.model_variant_id
            if (flow, variant) not in assignment.model_selected:
                raise ValueError("Warm start model is not available in the target")

            components = solution.components
            if (
                not components
                or len(components) > data.max_components
                or tuple(component.component_index for component in components)
                != tuple(range(1, len(components) + 1))
            ):
                raise ValueError("Warm start components must form an available prefix")

            selected_variables.append(assignment.model_selected[flow, variant])
            expected_layers = set(
                data.static_profiles_by_variant[
                    variant
                ].topology.layers_in_topological_order()
            )
            assigned_layers: set[str] = set()
            for component in components:
                index = component.component_index
                worker_key = flow, index, component.worker_id
                priority_key = (*worker_key, component.priority)
                if worker_key not in assignment.component_worker:
                    raise ValueError("Warm start worker is not available in the target")
                if priority_key not in compute_queue.component_priority:
                    raise ValueError(
                        "Warm start priority is not available in the target"
                    )

                selected_variables.extend(
                    (
                        assignment.component_active[flow, index],
                        assignment.component_worker[worker_key],
                        compute_queue.component_priority[priority_key],
                    )
                )
                for layer in component.layer_keys:
                    if layer not in expected_layers or layer in assigned_layers:
                        raise ValueError(
                            "Warm start contains unknown or repeated layers"
                        )
                    assigned_layers.add(layer)
                    selected_variables.append(
                        assignment.layer_component[flow, variant, layer, index]
                    )
            if assigned_layers != expected_layers:
                raise ValueError("Warm start must assign every selected model layer")
            selected_variables.append(assignment.component_last[flow, len(components)])

        native_model = formulation_model.native_model
        variables = native_model.getVars()
        native_model.setAttr(
            GRB.Attr.Start, variables, [GRB.UNDEFINED] * len(variables)
        )
        native_model.setAttr(
            GRB.Attr.Start, decision_variables, [0.0] * len(decision_variables)
        )
        native_model.setAttr(
            GRB.Attr.Start, selected_variables, [1.0] * len(selected_variables)
        )
        native_model.update()
