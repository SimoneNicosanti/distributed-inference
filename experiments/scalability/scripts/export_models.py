"""Explicit torchvision preparation: eager ONNX export then production optimization."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import onnx
import torch
from scalability_benchmark.catalog import CatalogEntry, artifact_hashes, file_digest
from scalability_benchmark.configuration import load_configuration
from scalability_benchmark.profiling.static import variant_metadata
from torchvision import models

from artifacts.contracts.artifact_workspace import ArtifactWorkspace
from model_manager.adapters.outbound.model_optimizer.onnx_model_optimizer import (
    OnnxModelOptimizer,
)
from model_manager.domain.model import ModelInfo
from shared.model.model import CNNArchitectureInfo, ModelId, ModelTask
from shared.model.model_variant import (
    ModelVariantId,
    PerformanceMetric,
    PerformanceMetricType,
)
from utils.model.optimization.api.optimization_level import OptimizationLevel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/default.yaml",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--opset", type=int)
    parser.add_argument("--weights", choices=("random", "local", "download"))
    parser.add_argument("--weights-directory", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help="Export only these configured model names",
    )
    args = parser.parse_args()
    config = load_configuration(args.config)
    selected = (
        set(args.models)
        if args.models
        else {model.name for model in config.catalog.models}
    )
    unknown = selected - {model.name for model in config.catalog.models}
    if unknown:
        parser.error(f"Unknown model names: {', '.join(sorted(unknown))}")
    args.seed = config.export.seed if args.seed is None else args.seed
    args.opset = config.export.opset if args.opset is None else args.opset
    args.weights = config.export.weights if args.weights is None else args.weights
    if args.seed < 0 or args.opset < 13:
        parser.error("Export seed must be nonnegative and opset must be at least 13")
    if args.weights == "local" and args.weights_directory is None:
        parser.error("--weights local requires --weights-directory")
    config.model_directory.mkdir(parents=True, exist_ok=True)
    raw_directory = config.model_directory / "source"
    raw_directory.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    for specification in config.catalog.models:
        if specification.name not in selected:
            continue
        destination = config.model_directory / specification.filename
        if destination.exists() and not args.force:
            raise FileExistsError(
                f"{destination} exists; --force explicitly replaces artifacts"
            )
        torch.manual_seed(args.seed)
        constructor = models.get_model_builder(specification.name)
        weights = (
            models.get_model_weights(specification.name).DEFAULT
            if args.weights == "download"
            else None
        )
        model = constructor(weights=weights)
        if args.weights == "local":
            assert args.weights_directory is not None
            state = torch.load(
                args.weights_directory / f"{specification.name}.pth",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state)
        model.eval()
        example = torch.zeros(specification.input_shape, dtype=torch.float32)
        raw_path = raw_directory / specification.filename
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (example,),
                str(raw_path),
                dynamo=False,
                opset_version=args.opset,
                input_names=["images"],
                output_names=["logits"],
                export_params=True,
                keep_initializers_as_inputs=False,
            )
        proto = onnx.load(str(raw_path))
        for index, node in enumerate(proto.graph.node):
            node.name = f"node_{index:06d}_{node.op_type}"
        onnx.save(proto, str(raw_path))
        onnx.checker.check_model(str(raw_path))
        entry = CatalogEntry(
            specification,
            raw_path,
            ModelVariantId(ModelId(specification.name), "fp32-batch1"),
            PerformanceMetric(
                PerformanceMetricType.ACCURACY, config.catalog.dataset, 0.0
            ),
            {},
        )
        workspace = ArtifactWorkspace(config.model_directory)
        # BASIC preserves standard ONNX operators; extended ORT fusion can produce
        # provider-specific operators unsupported by the existing static profiler.
        OnnxModelOptimizer().optimize_model(
            ArtifactWorkspace(raw_directory, raw_path),
            workspace,
            ModelInfo(ModelTask.CLASSIFICATION, CNNArchitectureInfo()),
            variant_metadata(entry).model_variant_info,
            OptimizationLevel.BASIC,
        )
        if workspace.entrypoint_path is None:
            raise RuntimeError("Production optimizer did not set its output path")
        if workspace.entrypoint_path.with_name(
            f"{workspace.entrypoint_path.name}.data"
        ).exists():
            raise ValueError(
                "Unexpected external data; keep original sidecar names when preparing this model manually"
            )
        workspace.entrypoint_path.replace(destination)
        final_model = onnx.load(str(destination))
        for index, node in enumerate(final_model.graph.node):
            node.name = f"node_{index:06d}_{node.op_type}"
        onnx.save(final_model, str(destination))
        onnx.checker.check_model(str(destination))
        manifest = {
            "model": specification.name,
            "input_shape": specification.input_shape,
            "seed": args.seed,
            "opset": args.opset,
            "export": "eager",
            "optimization": "BASIC",
            "weights": args.weights,
            "weights_identity": str(weights) if weights is not None else args.weights,
            "local_weights_sha256": file_digest(
                args.weights_directory / f"{specification.name}.pth"
            )
            if args.weights == "local"
            else None,
            "artifacts": artifact_hashes(destination),
            "versions": {
                name: version(name)
                for name in ("torch", "torchvision", "onnx", "onnxruntime")
            },
        }
        destination.with_suffix(".export.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
