from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
from export_common import finalize_model
from transformers import AutoConfig

## TODO: TO CHECK FOR CORRECT EXPORT AND FUSION


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"

TASK_MAP = {
    "base": "feature-extraction",
    "masked-lm": "fill-mask",
    "sequence-classification": "text-classification",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face BERT model to ONNX with Optimum"
    )
    parser.add_argument(
        "--model",
        default="google-bert/bert-base-uncased",
        help="Hugging Face model ID or local model directory",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_MAP),
        default="base",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--static",
        action="store_true",
        help="Disable Optimum's dynamic batch and sequence axes",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--optimize",
        choices=["O1", "O2", "O3", "O4"],
        default=None,
        help="Optional Optimum/ONNX Runtime optimization during export",
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use the Dynamo exporter; normally use it only for opset >= 18",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing export directory",
    )
    return parser.parse_args()


def sanitize_model_name(name: str) -> str:
    return name.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def resolve_optimum_cli() -> str:
    venv_cli = Path(sys.executable).with_name("optimum-cli")
    if venv_cli.exists():
        return str(venv_cli)

    cli = shutil.which("optimum-cli")
    if cli is None:
        raise RuntimeError(
            "optimum-cli was not found. Install compatible versions of "
            "optimum, optimum-onnx and transformers in the current environment."
        )
    return cli


def validate_export(export_dir: Path, expected_opset: int) -> Path:
    candidates = sorted(export_dir.glob("*.onnx"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one ONNX file in {export_dir}, found: "
            f"{[path.name for path in candidates]}"
        )

    model_path = candidates[0]
    onnx.checker.check_model(str(model_path))

    model = onnx.load(str(model_path), load_external_data=False)
    default_opset = next(
        (item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}),
        None,
    )
    if default_opset != expected_opset:
        raise RuntimeError(f"Expected ONNX opset {expected_opset}, got {default_opset}")

    return model_path


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if args.dynamo and args.opset < 18:
        raise ValueError("--dynamo should only be used with opset >= 18")
    if args.optimize == "O4" and not args.device.startswith("cuda"):
        raise ValueError("Optimum O4 requires --device cuda")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{sanitize_model_name(args.model)}"
        f"_{args.task}"
        f"_b{args.batch_size}"
        f"_s{args.sequence_length}"
        f"_{args.precision}"
        f"_opset{args.opset}"
    )
    export_dir = output_root / stem

    if export_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{export_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(export_dir)

    optimum_task = TASK_MAP[args.task]

    command = [
        resolve_optimum_cli(),
        "export",
        "onnx",
        "--model",
        args.model,
        "--task",
        optimum_task,
        "--framework",
        "pt",
        "--opset",
        str(args.opset),
        "--dtype",
        args.precision,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--sequence_length",
        str(args.sequence_length),
        # "--revision",
        # args.revision,
    ]

    if args.static:
        command.append("--no-dynamic-axes")
    if args.optimize is not None:
        command.extend(["--optimize", args.optimize])
    if args.dynamo:
        command.append("--dynamo")
    if args.cache_dir is not None:
        command.extend(["--cache_dir", str(args.cache_dir.resolve())])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.atol is not None:
        command.extend(["--atol", str(args.atol)])

    command.append(str(export_dir))

    print("$", " ".join(command))
    subprocess.run(command, check=True)

    model_path = validate_export(export_dir, args.opset)

    finalize_model(
        model_path,
        model_path.with_name(f"{model_path.stem}_optimized.onnx"),
        opset=args.opset,
        precision=args.precision,
        is_transformer=True,
    )

    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None,
        trust_remote_code=args.trust_remote_code,
    )
    num_heads = int(config.num_attention_heads)
    hidden_size = int(config.hidden_size)
    optimized_path = model_path.with_name(
        f"{model_path.stem}_transformer_optimized.onnx"
    )

    print(f"\nExported model: {model_path}")
    print("Dynamic axes:", "disabled" if args.static else "enabled by Optimum")
    print("\nTransformer optimizer command:")
    print(
        f"{sys.executable} -m onnxruntime.transformers.optimizer "
        f'--input "{model_path}" '
        f'--output "{optimized_path}" '
        f"--model_type bert "
        f"--num_heads {num_heads} "
        f"--hidden_size {hidden_size} "
        f"--opt_level 0 "
        f"--verbose"
    )


if __name__ == "__main__":
    main()
