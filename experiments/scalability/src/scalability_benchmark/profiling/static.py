"""Reuse production profiling and DTO mapping without storage services."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import onnx

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from controller.adapters.outbound.profile.mappers.model_variant_static_profile_mapper import (
    ModelVariantStaticProfileMapper,
)
from controller.domain.profile.model_variant_static_profile import (
    ModelVariantStaticProfile,
)
from integration.dto.model.model_dto import ModelDto
from integration.dto.model.profiled_model_variant_dto import ProfiledModelVariantDto
from model_manager.adapters.outbound.model_max_cut_size_computer.pulp_max_cut_size_computer import (
    PulpMaxCutSizeComputer,
)
from model_manager.adapters.outbound.model_metadata_store.mappers.profiled_model_variant_mapper import (
    ProfiledModelVariantMapper,
)
from model_manager.adapters.outbound.model_optimizer.onnx_model_optimizer import (
    OnnxModelOptimizer,
)
from model_manager.adapters.outbound.model_profiler.onnx_contraction_computer import (
    OnnxContractionComputer,
)
from model_manager.adapters.outbound.model_profiler.onnx_invariant_properties_computer import (
    OnnxInvariantPropertiesComputer,
)
from model_manager.adapters.outbound.model_profiler.onnx_model_topology_builder import (
    OnnxModelTopologyBuilder,
)
from model_manager.adapters.outbound.model_profiler.onnx_shape_profile_computer import (
    OnnxShapeProfileComputer,
)
from model_manager.application.default_model_profiler import DefaultModelProfiler
from model_manager.application.ports.outbound.model_profiler import ModelProfiler
from model_manager.domain.model import ModelInfo
from model_manager.domain.model_variant import (
    DynamicShapeValues,
    InputInfo,
    ModelVariant,
    ModelVariantInfo,
)
from model_manager.domain.profiled_model_variant import ProfiledModelVariant
from scalability_benchmark.catalog import CatalogEntry
from scalability_benchmark.persistence import atomic_json
from scalability_benchmark.randomness import fingerprint
from shared.artifact.artifact_ref import ArtifactRef
from shared.model.model import CNNArchitectureInfo, ModelTask
from shared.model.model_variant import (
    ModelVariantFormat,
    NumericPrecision,
    QuantizationType,
)


def build_profiler() -> DefaultModelProfiler:
    topology = OnnxModelTopologyBuilder()
    return DefaultModelProfiler(
        topology,
        OnnxInvariantPropertiesComputer(),
        OnnxShapeProfileComputer(PulpMaxCutSizeComputer()),
        OnnxContractionComputer(OnnxModelOptimizer(), topology),
    )


def variant_metadata(entry: CatalogEntry) -> ModelVariant:
    model = onnx.load_model(str(entry.path), load_external_data=False)
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    if len(inputs) != 1 or inputs[0].name != "images":
        raise ValueError(
            f"{entry.path}: production profiler requires one input named 'images'; re-export using scripts/export_models.py"
        )
    if inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(f"{entry.path}: expected FP32 image input")
    dimensions = inputs[0].type.tensor_type.shape.dim
    if len(dimensions) != len(entry.settings.input_shape):
        raise ValueError(f"{entry.path}: input rank differs from configured shape")
    shape: list[int | str] = []
    bindings: dict[str, DynamicShapeValues] = {}
    for dimension, expected in zip(dimensions, entry.settings.input_shape, strict=True):
        kind = dimension.WhichOneof("value")
        if kind == "dim_value":
            if dimension.dim_value != expected:
                raise ValueError(
                    f"{entry.path}: configured shape contradicts static ONNX dimensions"
                )
            shape.append(expected)
        elif kind == "dim_param" and dimension.dim_param:
            symbol = dimension.dim_param
            binding = DynamicShapeValues(frozenset({expected}))
            if symbol in bindings and bindings[symbol] != binding:
                raise ValueError(f"{entry.path}: inconsistent binding for {symbol}")
            bindings[symbol] = binding
            shape.append(symbol)
        else:
            raise ValueError(
                f"{entry.path}: anonymous dimension; use the preparation script"
            )
    return ModelVariant(
        entry.variant_id,
        ModelVariantInfo(
            precision=NumericPrecision.FP32,
            quantization=QuantizationType.NONE,
            accuracies=(entry.accuracy,),
            format=ModelVariantFormat.ONNX,
            inputs_info={"images": InputInfo(tuple(shape))},
            dynamic_dimension_values=bindings,
        ),
    )


@dataclass(frozen=True)
class StaticModel:
    entry: CatalogEntry
    profile: ModelVariantStaticProfile
    dto: ProfiledModelVariantDto


def dto_record(dto: ProfiledModelVariantDto) -> dict[str, object]:
    """Canonicalize unordered DTO tensors before hashing across processes."""
    record = dto.model_dump(mode="json")
    edges = record["profile"]["topology"]["edges"]
    for edge in edges:
        edge["tensors"] = sorted(edge["tensors"])
    edges.sort(key=lambda edge: (edge["source"], edge["target"]))
    return record


async def profile_entry(
    entry: CatalogEntry,
    cache_directory: Path,
    software_hash: str,
    profiler: ModelProfiler | None = None,
) -> StaticModel:
    info = ModelInfo(ModelTask.CLASSIFICATION, CNNArchitectureInfo())
    variant = variant_metadata(entry)
    cache_key = fingerprint(
        {
            "artifacts": entry.artifacts,
            "settings": entry.settings.model_dump(mode="json"),
            "accuracy": entry.accuracy.value,
            "dataset": entry.accuracy.dataset,
            "software": software_hash,
        }
    )
    cache_path = cache_directory / f"{cache_key}.json"
    if cache_path.is_file():
        dto = ProfiledModelVariantDto.model_validate_json(cache_path.read_text())
    else:
        profile = await (profiler or build_profiler()).profile_model_variant(
            ArtifactWorkspace(entry.path.parent, entry.path), info, variant
        )
        dto = ProfiledModelVariantMapper.to_profiled_model_variant_dto(
            ProfiledModelVariant(variant, profile, ArtifactRef(cache_key))
        )
        await asyncio.to_thread(atomic_json, cache_path, dto_record(dto))
    mapped = ModelVariantStaticProfileMapper.model_variant_static_profile_from_dto(
        dto,
        ModelDto(
            model_id=entry.variant_id.model_id,
            model_task=info.model_task,
            architecture_info=info.architecture_info,
        ),
    )
    if len(mapped.shape_profiles) != 1:
        raise ValueError(f"{entry.path}: optimizer requires exactly one shape profile")
    return StaticModel(entry, mapped, dto)
