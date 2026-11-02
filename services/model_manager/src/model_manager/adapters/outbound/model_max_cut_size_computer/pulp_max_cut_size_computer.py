import math
from typing import override

import pulp

from model_manager.application.ports.outbound.max_cut_size_computer import (
    MaxCutSizeComputer,
)
from model_manager.domain.model_variant_topology import ModelVariantTopology
from model_manager.domain.shape_profile import TensorShapeProperty
from shared.model.keys import TensorKey


class PulpMaxCutSizeComputer(MaxCutSizeComputer):
    @staticmethod
    @override
    def compute_model_variant_max_cut_size(
        model_variant_topology: ModelVariantTopology,
        tensor_shape_properties: dict[TensorKey, TensorShapeProperty],
    ) -> float:

        return PulpMaxCutSizeComputer._sync_compute_model_variant_max_cut_size(
            model_variant_topology,
            tensor_shape_properties,
        )

    @staticmethod
    def _sync_compute_model_variant_max_cut_size(
        model_variant_topology: ModelVariantTopology,
        tensor_shape_properties: dict[TensorKey, TensorShapeProperty],
    ) -> float:

        problem = pulp.LpProblem("max_cut", pulp.LpMaximize)

        c_dict = {
            tensor: problem.add_variable(name=f"c_{tensor}", cat="Binary")
            for tensor in tensor_shape_properties
        }
        y_dict = {
            layer: problem.add_variable(name=f"y_{layer}", cat="Binary")
            for layer in model_variant_topology.layers().keys()
        }

        for edge in model_variant_topology.edges().keys():
            source, dest = edge[0], edge[1]
            problem += y_dict[dest] <= y_dict[source]

        for tensor in tensor_shape_properties:
            source = model_variant_topology.find_tensor_producer(tensor)
            if source is None:
                raise KeyError()
            problem += c_dict[tensor] <= y_dict[source]
            consumer_set = model_variant_topology.find_tensor_consumer_set(tensor)

            problem += c_dict[tensor] <= sum(
                1 - y_dict[consumer] for consumer in consumer_set
            )

        sizes = {
            tensor: math.prod(shape_property.shape) * 4 / (1024 * 1024)
            for tensor, shape_property in tensor_shape_properties.items()
        }

        problem += pulp.lpSum(size * c_dict[tensor] for tensor, size in sizes.items())

        for mip in (True, False):
            try:
                solved = problem.solve(pulp.HiGHS(msg=False, mip=mip, timeLimit=30))
            except pulp.PulpSolverError:
                continue
            if (
                solved.status == pulp.LpSolveStatus.Optimal
                and solved.objective is not None
                and math.isfinite(solved.objective)
            ):
                return float(solved.objective)

        return sum(sizes.values())
