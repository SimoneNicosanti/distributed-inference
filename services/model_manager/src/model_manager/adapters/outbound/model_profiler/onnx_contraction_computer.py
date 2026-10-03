from pathlib import Path
from tempfile import TemporaryDirectory
from typing import override

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.application.ports.outbound.contraction_computer import (
    ContractionComputer,
)
from model_manager.application.ports.outbound.model_optimizer import ModelOptimizer
from model_manager.application.ports.outbound.model_topology_builder import (
    ModelTopologyBuilder,
)
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import ModelVariantInfo
from model_manager.domain.model_variant_topology import ModelVariantTopology
from shared.model.keys import LayerKey, TensorKey
from utils.model.optimization.api.optimization_level import OptimizationLevel


class OnnxContractionComputer(ContractionComputer):
    def __init__(
        self,
        model_optimizer: ModelOptimizer,
        model_topology_builder: ModelTopologyBuilder,
    ) -> None:
        self._model_optimizer = model_optimizer
        self._model_topology_builder = model_topology_builder

    @override
    def compute_contractions(
        self,
        artifact_workspace: ArtifactWorkspace,
        model_info: ModelInfo,
        model_variant_info: ModelVariantInfo,
        topology: ModelVariantTopology,
    ) -> list[tuple[LayerKey, ...]]:
        with TemporaryDirectory(prefix="model-contractions-") as temporary_directory:
            optimized_workspace = ArtifactWorkspace(root_path=Path(temporary_directory))
            self._model_optimizer.optimize_model(
                input_workspace=artifact_workspace,
                output_workspace=optimized_workspace,
                model_info=model_info,
                model_variant_info=model_variant_info,
                opt_level=OptimizationLevel.EXTENDED,
            )

            if optimized_workspace.entrypoint_path is None:
                raise RuntimeError("Model optimizer did not set the output entrypoint")

            optimized_topology = self._model_topology_builder.build_model_topology(
                optimized_workspace,
                model_info,
                model_variant_info,
            )
            return self._extract_contractions(optimized_topology, topology)

    @staticmethod
    def _extract_contractions(
        optimized_topology: ModelVariantTopology,
        topology: ModelVariantTopology,
    ) -> list[tuple[LayerKey, ...]]:
        topological_order = topology.get_topological_sort()
        layer_positions = {
            layer_key: position for position, layer_key in enumerate(topological_order)
        }

        contractions: set[tuple[LayerKey, ...]] = set()
        for layer_key, layer_description in optimized_topology.layers().items():
            if layer_description.is_input or layer_description.is_output:
                continue

            inputs, outputs = (
                optimized_topology.extract_incoming_outgoing_tensors_of_partition(
                    {layer_key}
                )
            )

            contracted_layers = OnnxContractionComputer._infer_contracted_group(
                topology,
                inputs,
                outputs,
            )
            if contracted_layers is None or len(contracted_layers) < 2:
                continue

            contractions.add(
                tuple(
                    layer_key
                    for layer_key in topological_order
                    if layer_key in contracted_layers
                )
            )

        ordered_contractions = sorted(
            contractions,
            key=lambda contraction: layer_positions[contraction[0]],
        )

        assigned_layers: set[LayerKey] = set()
        for contraction in ordered_contractions:
            overlapping_layers = assigned_layers.intersection(contraction)
            if overlapping_layers:
                raise ValueError(
                    "Optimized nodes produce overlapping contractions for layers "
                    f"{sorted(overlapping_layers)!r}"
                )
            assigned_layers.update(contraction)

        return ordered_contractions

    @staticmethod
    def _infer_contracted_group(
        topology: ModelVariantTopology,
        optimized_inputs: set[TensorKey],
        optimized_outputs: set[TensorKey],
    ) -> set[LayerKey] | None:
        if not optimized_outputs:
            return None

        output_producers = {
            topology.find_tensor_producer(tensor_key)
            for tensor_key in optimized_outputs
        }
        if None in output_producers:
            return None

        pending = [producer for producer in output_producers if producer is not None]
        contracted_layers: set[LayerKey] = set()

        while pending:
            layer_key = pending.pop()
            if layer_key in contracted_layers:
                continue

            layer_description = topology.get_layer_description(layer_key)
            if layer_description.is_input or layer_description.is_output:
                return None

            contracted_layers.add(layer_key)

            for (source, _), edge_description in topology.get_layer_in_edges(
                layer_key
            ).items():
                internal_tensors = edge_description.tensors - optimized_inputs
                if internal_tensors:
                    pending.append(source)

        incoming, outgoing = topology.extract_incoming_outgoing_tensors_of_partition(
            contracted_layers
        )
        if incoming != optimized_inputs or outgoing != optimized_outputs:
            return None

        return contracted_layers
