import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import override

from artifacts.workspace.artifact_workspace import (
    ArtifactWorkspace,
)
from integration.model.model import ModelInfo
from integration.model.model_version import (
    ModelVersion,
    ModelVersionInfo,
)
from model_manager.application.ports.outbound.model_graph_extractor import (
    ModelGraphExtractor,
)
from model_manager.application.ports.outbound.model_optimizer import (
    ModelOptimizer,
)
from model_manager.application.ports.outbound.model_profiler import (
    ModelProfiler,
)
from model_manager.domain.model_version_graph import (
    ModelVersionGraph,
)
from model_manager.domain.profiled_model_version import ProfiledModelVersion
from utils.model.optimization.api.optimization_level import (
    OptimizationLevel,
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
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_version: ModelVersion,
    ) -> ProfiledModelVersion:
        ## The whole profiling can be CPU bound, so we wrap it into
        ## a thread in order not to block the coroutine execution
        model_version_graph = await asyncio.to_thread(
            self._profile_model_sync,
            artifact_workspace,
            model_info,
            model_version.model_version_info,
        )

        return ProfiledModelVersion(
            model_version=model_version,
            model_version_graph=model_version_graph,
        )

    def _profile_model_sync(
        self,
        input_model_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_version_info: ModelVersionInfo,
    ) -> ModelVersionGraph:
        with TemporaryDirectory() as tmp_path:
            basic_opt_model_workspace = ArtifactWorkspace(
                root_path=Path(tmp_path) / "basic"
            )
            self._model_optimizer.optimize_model(
                input_model_workspace,
                basic_opt_model_workspace,
                model_info,
                model_version_info,
                OptimizationLevel.BASIC,
            )

            basic_model_graph = self._model_graph_extractor.extract_model_graph(
                basic_opt_model_workspace,
                model_version_info,
                profile_flops=True,
                profile_tensors=True,
            )

            ext_opt_model_workspace = ArtifactWorkspace(
                root_path=Path(tmp_path) / "extended"
            )
            self._model_optimizer.optimize_model(
                input_model_workspace,
                ext_opt_model_workspace,
                model_info,
                model_version_info,
                OptimizationLevel.EXTENDED,
            )

            ext_model_graph = self._model_graph_extractor.extract_model_graph(
                ext_opt_model_workspace,
                model_version_info,
                profile_flops=False,
                profile_tensors=False,
            )

        agg_model_graph = self._model_graph_extractor.aggregate_model_graphs(
            basic_model_graph, ext_model_graph
        )

        return agg_model_graph
