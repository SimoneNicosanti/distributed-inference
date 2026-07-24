from distributed_inference.application.model_profile.contracts.model_profile import (
    ModelProfile,
)
from distributed_inference.domain.identifiers import ModelVersionId
from distributed_inference.domain.model_graph_info import ModelGraph, ModelInfo

from distributed_inference.application.model_artifact.contracts.model_version_materializer import (
    ModelMaterializer,
)

from distributed_inference.application.model_metadata_store.contract.model_metadata_store import (
    ModelMetadataStore,
)


from distributed_inference.application.model_profile.profiling.model_profile_agg import (
    profile_with_model_optimization,
    compute_aggregate_model_graph,
)

from distributed_inference.application.model_profile.optimization.model_optimize import (
    OptimizationLevel,
)


class DefaultModelProfile(ModelProfile):
    def __init__(
        self,
        model_materializer: ModelMaterializer,
        model_metadata_store: ModelMetadataStore,
    ):
        self.model_materializer = model_materializer
        self.model_metadata_store = model_metadata_store
        pass

    def profile_model(
        self,
        model_version_id: ModelVersionId,
    ) -> ModelGraph:
        return self.profile_model_with_aggregation(model_version_id)

    def profile_model_with_no_optimization(
        self,
        model_version_id: ModelVersionId,
    ) -> ModelGraph:
        model_info: ModelInfo = self.model_metadata_store.get_model_info(
            model_version_id
        )

        with self.model_materializer.materialize_model(model_version_id) as model_path:
            base_model_graph = profile_with_model_optimization(
                model_path, model_info, OptimizationLevel.BASIC
            )

            return base_model_graph
        pass

    def profile_model_with_optimization(
        self,
        model_version_id: ModelVersionId,
    ) -> ModelGraph:

        model_info: ModelInfo = self.model_metadata_store.get_model_info(
            model_version_id
        )

        with self.model_materializer.materialize_model(model_version_id) as model_path:
            ext_model_graph = profile_with_model_optimization(
                model_path, model_info, OptimizationLevel.EXTENDED
            )

            return ext_model_graph
        pass

    def profile_model_with_aggregation(
        self,
        model_version_id: ModelVersionId,
    ) -> ModelGraph:

        model_info: ModelInfo = self.model_metadata_store.get_model_info(
            model_version_id
        )

        with self.model_materializer.materialize_model(model_version_id) as model_path:
            base_model_graph = profile_with_model_optimization(
                model_path, model_info, OptimizationLevel.BASIC
            )

            ext_model_graph = profile_with_model_optimization(
                model_path, model_info, OptimizationLevel.EXTENDED
            )

            agg_model_graph = compute_aggregate_model_graph(
                base_model_graph, ext_model_graph
            )

            return agg_model_graph
