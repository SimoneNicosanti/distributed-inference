import asyncio
from typing import override

from artifacts.contracts.artifact_workspace import (
    ArtifactWorkspace,
)
from model_manager.application.ports.outbound.contraction_computer import (
    ContractionComputer,
)
from model_manager.application.ports.outbound.invariant_properties_computer import (
    InvariantPropertiesComputer,
)
from model_manager.application.ports.outbound.model_profiler import (
    ModelProfiler,
)
from model_manager.application.ports.outbound.model_topology_builder import (
    ModelTopologyBuilder,
)
from model_manager.application.ports.outbound.shape_profile_computer import (
    ShapeProfileComputer,
)
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import (
    ModelVariant,
    ModelVariantInfo,
)
from model_manager.domain.model_variant_profile import (
    ModelVariantProfile,
)


## TODO: Here the default profiler should not get ModelOptimizer
## It should get a registry of Optimizers and Profilers with
## Different classes based on model formats (ONNX vs TF)
class DefaultModelProfiler(ModelProfiler):
    def __init__(
        self,
        model_topology_builder: ModelTopologyBuilder,
        invariant_properties_computer: InvariantPropertiesComputer,
        shape_profile_computer: ShapeProfileComputer,
        contraction_computer: ContractionComputer,
    ) -> None:
        self._model_topology_builder = model_topology_builder
        self._invariant_properties_computer = invariant_properties_computer
        self._shape_profile_computer = shape_profile_computer
        self._contraction_computer = contraction_computer

    @override
    async def profile_model_variant(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant: ModelVariant,
    ) -> ModelVariantProfile:
        ## The whole profiling can be CPU bound, so we wrap it into
        ## a thread in order not to block the coroutine execution
        model_variant_profile = await asyncio.to_thread(
            self._profile_model_sync,
            artifact_workspace,
            model_info,
            model_variant.model_variant_info,
        )

        return model_variant_profile

    def _profile_model_sync(
        self,
        input_model_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
    ) -> ModelVariantProfile:

        topology = self._model_topology_builder.build_model_topology(
            input_model_workspace,
            model_info,
            model_variant_info,
        )
        invariant_layer_properties = (
            self._invariant_properties_computer.compute_layer_invariant_properties(
                input_model_workspace, model_info, model_variant_info, topology
            )
        )
        invariant_tensor_properties = (
            self._invariant_properties_computer.compute_tensor_invariant_properties(
                input_model_workspace, model_info, model_variant_info, topology
            )
        )
        shape_profiles = self._shape_profile_computer.compute_shape_profiles(
            input_model_workspace, model_info, model_variant_info, topology
        )

        optimization_contractions = self._contraction_computer.compute_contractions(
            input_model_workspace, model_info, model_variant_info, topology
        )

        return ModelVariantProfile(
            topology=topology,
            shape_profiles=shape_profiles,
            invariant_layer_properties=invariant_layer_properties,
            invariant_tensor_properties=invariant_tensor_properties,
            optimization_contractions=optimization_contractions,
        )
