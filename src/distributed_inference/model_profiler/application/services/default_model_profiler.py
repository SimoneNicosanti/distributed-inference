import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import override

from distributed_inference.artifact_materializer.domain.materialized_artifact import (
    MaterializedArtifact,
)
from distributed_inference.artifact_processing.artifact_workspace import (
    ArtifactWorkspace,
)
from distributed_inference.model_manager.domain.model import ModelInfo
from distributed_inference.model_manager.domain.model_version import (
    ModelVersion,
    ModelVersionInfo,
    ProfiledModelVersion,
)
from distributed_inference.model_manager.domain.model_version_graph import (
    ModelVersionGraph,
)
from distributed_inference.model_optimizer.application.ports.outbound.model_optimizer import (
    ModelOptimizer,
)
from distributed_inference.model_optimizer.domain.optimization_level import (
    OptimizationLevel,
)
from distributed_inference.model_profiler.application.ports.inbound.model_profiler import (
    ModelProfiler,
)
from distributed_inference.model_profiler.application.ports.outbound.model_graph_extractor import (
    ModelGraphExtractor,
)


## TODO: Here the default profiler should not get ModelOptimizer
## It should get a registry of Optimizers and Profilers with
## Different classes based on model formats (ONNX vs TF)
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
    async def profile_model_version(
        self,
        artifact_concrete_paths: MaterializedArtifact,
        model_info: ModelInfo,
        model_version: ModelVersion,
    ) -> ProfiledModelVersion:
        ## The whole profiling can be CPU bound, so we wrap it into
        ## a thread in order not to block the coroutine execution
        model_version_graph = await asyncio.to_thread(
            self._profile_model_sync,
            artifact_concrete_paths,
            model_info,
            model_version.model_version_info,
        )

        return ProfiledModelVersion(
            model_version_id=model_version.model_version_id,
            model_version_info=model_version.model_version_info,
            model_version_graph=model_version_graph,
        )

    def _profile_model_sync(
        self,
        artifact_concrete_paths: MaterializedArtifact,
        model_info: ModelInfo,
        model_version_info: ModelVersionInfo,
    ) -> ModelVersionGraph:
        with TemporaryDirectory() as tmp_path:
            basic_concrete_paths = ArtifactWorkspace(root_path=Path(tmp_path))
            self._model_optimizer.optimize_model(
                artifact_concrete_paths,
                basic_concrete_paths,
                model_info,
                model_version_info,
                OptimizationLevel.BASIC,
            )

            basic_model_graph = self._model_graph_extractor.extract_model_graph(
                basic_concrete_paths,
                model_version_info,
                profile_flops=True,
                profile_tensors=True,
            )

            ext_concrete_paths = ArtifactWorkspace(root_path=Path(tmp_path))
            self._model_optimizer.optimize_model(
                artifact_concrete_paths,
                ext_concrete_paths,
                model_info,
                model_version_info,
                OptimizationLevel.EXTENDED,
            )

            ext_model_graph = self._model_graph_extractor.extract_model_graph(
                ext_concrete_paths,
                model_version_info,
                profile_flops=False,
                profile_tensors=False,
            )

        agg_model_graph = self._model_graph_extractor.aggregate_model_graphs(
            basic_model_graph, ext_model_graph
        )

        return agg_model_graph
