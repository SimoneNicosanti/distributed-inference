"""Local artifact discovery and seed-stable model identities and accuracies."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import onnx

from scalability_benchmark.configuration import ExperimentConfig, ModelSettings
from scalability_benchmark.randomness import stream
from shared.model.model import ModelId
from shared.model.model_variant import (
    ModelVariantId,
    PerformanceMetric,
    PerformanceMetricType,
)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CatalogEntry:
    settings: ModelSettings
    path: Path
    variant_id: ModelVariantId
    accuracy: PerformanceMetric
    artifacts: dict[str, str]


def synthetic_accuracies(
    config: ExperimentConfig, seed: int
) -> dict[str, PerformanceMetric]:
    low, high = config.catalog.accuracy_range
    ranked = sorted(config.catalog.models, key=lambda item: item.accuracy_rank)
    return {
        model.name: PerformanceMetric(
            type=PerformanceMetricType.ACCURACY,
            dataset=config.catalog.dataset,
            value=low
            + (rank + stream(seed, "accuracy", model.name).uniform(0.1, 0.9))
            * (high - low)
            / len(ranked),
        )
        for rank, model in enumerate(ranked)
    }


def selection_order(config: ExperimentConfig, seed: int) -> tuple[str, ...]:
    names = sorted(model.name for model in config.catalog.models)
    stream(seed, "model-selection").shuffle(names)
    return tuple(names)


def artifact_hashes(path: Path) -> dict[str, str]:
    model = onnx.load_model(str(path), load_external_data=False)
    paths = {path.resolve()}
    for tensor in onnx.external_data_helper._get_all_tensors(model):
        if not onnx.external_data_helper.uses_external_data(tensor):
            continue
        location = next(
            (item.value for item in tensor.external_data if item.key == "location"),
            None,
        )
        if location is None:
            raise ValueError(f"Missing external-data location in {path}")
        external = (path.parent / location).resolve()
        if not external.is_relative_to(path.parent.resolve()):
            raise ValueError(f"External data escapes model directory: {location}")
        paths.add(external)
    return {
        item.relative_to(path.parent.resolve()).as_posix(): file_digest(item)
        for item in sorted(paths)
    }


def discover_catalog(config: ExperimentConfig, seed: int) -> tuple[CatalogEntry, ...]:
    missing = [
        model.filename
        for model in config.catalog.models
        if not (config.model_directory / model.filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing local ONNX models in {config.model_directory}: {', '.join(missing)}. "
            "Prepare them explicitly with scripts/export_models.py; see README.md."
        )
    accuracies = synthetic_accuracies(config, seed)
    by_name = {model.name: model for model in config.catalog.models}
    return tuple(
        CatalogEntry(
            settings=by_name[name],
            path=config.model_directory / by_name[name].filename,
            variant_id=ModelVariantId(ModelId(name), "fp32-batch1"),
            accuracy=accuracies[name],
            artifacts=artifact_hashes(config.model_directory / by_name[name].filename),
        )
        for name in selection_order(config, seed)
    )
