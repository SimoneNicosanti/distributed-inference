from typing import override

from controller.application.deployment.abc.model_partition_build_manager import (
    ModelPartitionBuildManager,
)
from controller.application.ports.outbound.partition.partition_builder_gateway import (
    PartitionBuilderGateway,
)
from controller.domain.optimization.solving.optimization_solution import (
    OptimizationSolution,
)
from shared.model.model_partition import ModelPartitionId


class DefaultModelPartitionBuildManager(ModelPartitionBuildManager):
    def __init__(self, partition_builder_gateway: PartitionBuilderGateway) -> None:
        super().__init__()
        self._partition_builder_gateway: PartitionBuilderGateway = (
            partition_builder_gateway
        )

    @override
    async def build_model_partitions_from_solution(
        self, solution: OptimizationSolution
    ) -> None:

        model_partition_ids = self._build_all_model_partition_ids_from_solution(
            solution,
        )
        await self._partition_builder_gateway.build_all_partitions_by_ids(
            model_partition_ids,
        )

    @staticmethod
    def _build_all_model_partition_ids_from_solution(
        solution: OptimizationSolution,
    ) -> set[ModelPartitionId]:

        ids = set()
        for flow_solution in solution.flow_solutions:
            model_variant_id = flow_solution.model_variant_id

            for component in flow_solution.components:
                layer_keys = component.executable_layer_keys
                if not layer_keys:
                    continue

                model_partition_id = ModelPartitionId(
                    model_variant_id=model_variant_id,
                    layers=layer_keys,
                )
                ids.add(model_partition_id)
        return ids
