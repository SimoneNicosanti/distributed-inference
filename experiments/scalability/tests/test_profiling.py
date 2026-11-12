from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from scalability_benchmark.catalog import CatalogEntry, synthetic_accuracies
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.profiling import static
from scalability_benchmark.profiling.execution import execution_profile
from scalability_benchmark.profiling.memory import estimate_memory
from scalability_benchmark.profiling.static import StaticModel, profile_entry

from controller.domain.profile.model_variant_execution_profile import Backend
from model_manager.adapters.outbound.model_metadata_store.mappers.profiled_model_variant_mapper import (
    ProfiledModelVariantMapper,
)
from model_manager.application.ports.outbound.model_profiler import ModelProfiler
from model_manager.domain.model_variant import InputInfo, ModelVariant, ModelVariantInfo
from shared.model.model import ModelId
from shared.model.model_variant import (
    ModelVariantFormat,
    ModelVariantId,
    NumericPrecision,
    QuantizationType,
)


@pytest.fixture
def model(
    config: ExperimentConfig,
    make_model: Callable[[CatalogEntry], StaticModel],
    tmp_path: Path,
) -> StaticModel:
    settings = config.catalog.models[0]
    return make_model(
        CatalogEntry(
            settings,
            tmp_path / settings.filename,
            ModelVariantId(ModelId(settings.name), "fp32-batch1"),
            synthetic_accuracies(config, 42)[settings.name],
            {},
        )
    )


def test_liveness_releases_consumed_activations(model: StaticModel) -> None:
    memory = estimate_memory(model.profile)
    assert memory.persistent_weights_bytes == 8 * 8 * 4
    assert memory.peak_live_activations_bytes == 2 * 8 * 4
    assert memory.summed_activations_bytes == 4 * 8 * 4
    partition = estimate_memory(model.profile, ("conv", "relu"))
    assert partition.peak_live_activations_bytes < partition.summed_activations_bytes


@pytest.mark.parametrize("tier", ["device", "edge", "cloud"])
def test_roofline_uses_operator_efficiency_and_stable_jitter(
    model: StaticModel, config: ExperimentConfig, tier: str
) -> None:
    settings = config.execution.model_copy(update={"jitter_fraction": 0.0})
    hardware = config.hardware[tier]
    result = execution_profile(model.profile, hardware, settings, 42, tier)
    assert (
        result.profile.shape_execution_profiles[0].input_shapes
        == model.profile.shape_profiles[0].input_shapes
    )
    backend = result.profile.shape_execution_profiles[0].backend_profiles[0]
    assert backend.backend == Backend(hardware.backend)
    convolution = settings.groups["convolution"]
    expected = hardware.overhead_s + max(
        1024 / (hardware.compute_flops_s * convolution.compute),
        (256 + 64) / (hardware.memory_bytes_s * convolution.memory),
    )
    assert backend.layer_profiles["conv"].execution_time_s == expected
    assert backend.layer_profiles["input"].execution_time_s == 0
    assert result.fallback_operators == ()
    first = execution_profile(model.profile, hardware, config.execution, 42, tier)
    assert first == execution_profile(
        model.profile, hardware, config.execution, 42, tier
    )
    for layer in ("conv", "relu", "pool"):
        value = (
            first.profile.shape_execution_profiles[0]
            .backend_profiles[0]
            .layer_profiles[layer]
            .execution_time_s
        )
        base = backend.layer_profiles[layer].execution_time_s
        assert 0.9 * base <= value <= 1.1 * base


@pytest.mark.asyncio
async def test_static_profile_uses_injected_production_port_and_cache(
    model: StaticModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variant = ModelVariant(
        model.entry.variant_id,
        ModelVariantInfo(
            precision=NumericPrecision.FP32,
            quantization=QuantizationType.NONE,
            accuracies=(model.entry.accuracy,),
            format=ModelVariantFormat.ONNX,
            inputs_info={"images": InputInfo((1, 8))},
            dynamic_dimension_values={},
        ),
    )
    monkeypatch.setattr(static, "variant_metadata", lambda entry: variant)
    manager_profile = ProfiledModelVariantMapper.from_profiled_model_variant_dto(
        model.dto
    ).profile
    profiler = AsyncMock(spec=ModelProfiler)
    profiler.profile_model_variant.return_value = manager_profile
    first = await profile_entry(model.entry, tmp_path, "software", profiler)
    second = await profile_entry(model.entry, tmp_path, "software", profiler)
    assert static.dto_record(first.dto) == static.dto_record(second.dto)
    profiler.profile_model_variant.assert_awaited_once()
