import math
from typing import override

import pulp

from model_manager.application.ports.outbound.max_cut_size_computer import (
    MaxCutSizeComputer,
)
from model_manager.domain.input_shape_invariable_properties import (
    InputShapeInvariableTensorProperties,
)
from model_manager.domain.input_shape_variable_properties import (
    InputShapeVariableTensorProperties,
)
from model_manager.domain.model_variant_topology import ModelVariantTopology
from shared.model.keys import TensorKey

BYTES_PER_MIB = 1024 * 1024


class PulpMaxCutSizeComputer(MaxCutSizeComputer):
    @staticmethod
    @override
    def compute_model_variant_max_cut_size(
        model_variant_topology: ModelVariantTopology,
        invariable_tensor_properties: dict[
            TensorKey, InputShapeInvariableTensorProperties
        ],
        variable_tensor_properties: dict[TensorKey, InputShapeVariableTensorProperties],
    ) -> float:

        return PulpMaxCutSizeComputer._sync_compute_model_variant_max_cut_size(
            model_variant_topology,
            invariable_tensor_properties,
            variable_tensor_properties,
        )

    @staticmethod
    def _sync_compute_model_variant_max_cut_size(
        model_variant_topology: ModelVariantTopology,
        invariable_tensor_properties: dict[
            TensorKey, InputShapeInvariableTensorProperties
        ],
        variable_tensor_properties: dict[TensorKey, InputShapeVariableTensorProperties],
    ) -> float:
        problem, fallback = PulpMaxCutSizeComputer._build_cut_problem(
            model_variant_topology,
            invariable_tensor_properties,
            variable_tensor_properties,
        )
        return PulpMaxCutSizeComputer._solve_cut_problem(problem, fallback=fallback)

    @staticmethod
    def _build_cut_problem(
        model_variant_topology: ModelVariantTopology,
        invariable_tensor_properties: dict[
            TensorKey, InputShapeInvariableTensorProperties
        ],
        variable_tensor_properties: dict[TensorKey, InputShapeVariableTensorProperties],
    ) -> tuple[pulp.LpProblem, float]:
        layers = model_variant_topology.layers()
        problem = pulp.LpProblem(
            "max_cut",
            pulp.LpMaximize,
        )

        c_dict = {
            tensor: problem.add_variable(name=f"c_{tensor}", cat="Binary")
            for tensor in invariable_tensor_properties
        }
        y_dict = {
            layer: problem.add_variable(name=f"y_{layer}", cat="Binary")
            for layer in layers
        }

        for edge in model_variant_topology.edges():
            source, dest = edge[0], edge[1]
            problem += y_dict[dest] <= y_dict[source]

        for tensor in invariable_tensor_properties:
            producer = model_variant_topology.find_tensor_producer(tensor)
            if producer is None:
                raise KeyError()
            problem += c_dict[tensor] <= y_dict[producer]
            consumer_set = model_variant_topology.find_tensor_consumer_set(tensor)

            problem += c_dict[tensor] <= sum(
                1 - y_dict[consumer] for consumer in consumer_set
            )

        sizes_mib = {}
        for tensor, shape_property in variable_tensor_properties.items():
            tensor_elems_num = math.prod(shape_property.shape)
            datasize_in_bytes = invariable_tensor_properties[
                tensor
            ].data_type_size_in_bytes
            sizes_mib[tensor] = tensor_elems_num * datasize_in_bytes / BYTES_PER_MIB

        problem += pulp.lpSum(
            size * c_dict[tensor] for tensor, size in sizes_mib.items()
        )
        return problem, sum(sizes_mib.values())

    @staticmethod
    def _solve_cut_problem(problem: pulp.LpProblem, *, fallback: float) -> float:
        for mip in (True, False):
            try:
                solved = problem.solve(pulp.HiGHS(msg=True, mip=mip, timeLimit=30))
            except pulp.PulpSolverError:
                continue
            if (
                solved.status == pulp.LpSolveStatus.Optimal
                and solved.objective is not None
                and math.isfinite(solved.objective)
            ):
                return float(solved.objective)

        return fallback
