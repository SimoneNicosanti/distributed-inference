from pathlib import Path
from tempfile import TemporaryDirectory
from typing import override

from distributed_inference.application.model_artifact.domain.artifact_bundle import (
    ArtifactConcretePaths,
)
from distributed_inference.application.model_optimize.contracts.model_optimizer import (
    ModelOptimizer,
    OptimizationLevel,
)
from distributed_inference.application.model_profile.contracts.model_graph_extractor import (
    ModelGraphExtractor,
)
from distributed_inference.application.model_profile.contracts.model_profiler import (
    ModelProfiler,
)
from distributed_inference.domain.model_graph_info import ModelGraph, ModelInfo


class DefaultModelProfiler(ModelProfiler):
    def __init__(
        self,
        model_optimizer: ModelOptimizer,
        model_graph_extractor: ModelGraphExtractor,
    ) -> None:
        self._model_optimizer = model_optimizer
        self._model_graph_extractor = model_graph_extractor
        pass

    @override
    def profile_model(
        self,
        artifact_concrete_paths: ArtifactConcretePaths,
        model_info: ModelInfo,
    ) -> ModelGraph:

        with TemporaryDirectory() as tmp_path:
            basic_concrete_paths = ArtifactConcretePaths(root_path=Path(tmp_path))
            self._model_optimizer.optimize_model(
                artifact_concrete_paths,
                basic_concrete_paths,
                model_info,
                OptimizationLevel.BASIC,
            )

            basic_model_graph = self._model_graph_extractor.extract_model_graph(
                basic_concrete_paths,
                model_info,
                profile_flops=True,
                profile_tensors=True,
            )

            ext_concrete_paths = ArtifactConcretePaths(root_path=Path(tmp_path))
            self._model_optimizer.optimize_model(
                artifact_concrete_paths,
                ext_concrete_paths,
                model_info,
                OptimizationLevel.EXTENDED,
            )

            ext_model_graph = self._model_graph_extractor.extract_model_graph(
                ext_concrete_paths,
                model_info,
                profile_flops=False,
                profile_tensors=False,
            )

        agg_model_graph = self._model_graph_extractor.aggregate_model_graphs(
            basic_model_graph, ext_model_graph
        )

        return agg_model_graph
