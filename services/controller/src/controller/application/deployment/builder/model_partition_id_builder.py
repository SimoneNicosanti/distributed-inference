from controller.domain.optimization.optimization_solution import OptimizationSolution
from shared.model.model_partition import ModelPartitionId


class ModelPartitionIdBuilder:
    @staticmethod
    def build_all_model_partition_ids_from_solution(
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
