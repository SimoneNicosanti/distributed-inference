from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import onnx
from transformers import AutoConfig
from export_common import finalize_model


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models"

## TODO: EXPORT TO BE FIXED FOR ATTENTION FUSION AND FOR DYNAMIC SIZES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face ViT model to ONNX with Optimum"
    )
    parser.add_argument(
        "--model",
        default="google/vit-base-patch16-224",
        help="Hugging Face model ID or local model directory",
    )
    parser.add_argument(
        "--task",
        default="image-classification",
        help="Optimum export task",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument(
        "--static",
        action="store_true",
        help="Disable Optimum's dynamic batch axis",
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
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
        help=(
            "Attention implementation to force at export time. 'eager' is the "
            "default because onnxruntime's transformer attention-fusion patterns "
            "are built for the classic (unfused) attention graph; models exported "
            "with 'sdpa' typically fail attention fusion."
        ),
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
    parser.add_argument(
        "--skip-attention-fusion",
        action="store_true",
        help="Skip running the onnxruntime transformer attention-fusion pass",
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


def resolve_image_configuration(
    model_name: str,
    revision: str,
    cache_dir: Path | None,
    trust_remote_code: bool,
    height: int | None,
    width: int | None,
    num_channels: int | None,
) -> tuple[object, int, int, int]:
    config = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=str(cache_dir.resolve()) if cache_dir else None,
        trust_remote_code=trust_remote_code,
    )

    image_size = getattr(config, "image_size", 224)
    if isinstance(image_size, (tuple, list)):
        if len(image_size) != 2:
            raise ValueError(f"Unsupported image_size: {image_size!r}")
        default_height = int(image_size[0])
        default_width = int(image_size[1])
    else:
        default_height = default_width = int(image_size)

    resolved_height = height or default_height
    resolved_width = width or default_width
    resolved_channels = num_channels or int(getattr(config, "num_channels", 3))

    return config, resolved_height, resolved_width, resolved_channels


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


def patch_scale_mul_to_div(input_path: Path, output_path: Path) -> int:
    """Rewrite `Mul(QK^T, 1/sqrt(d))` into an equivalent `Div(QK^T, sqrt(d))` node
    right before each Softmax.

    Why this is needed: transformers' shared `eager_attention_forward` (used by
    ViT and most current architectures) computes
        attn_weights = matmul(q, k.T) * scaling
    i.e. a Mul. onnxruntime's FusionAttention only recognizes a Mul-scaled QK
    pattern when it's paired with an attention-mask Add node (see qk_paths in
    fusion_attention.py: "path2" and "sdpa" both require Add before Mul/Sqrt).
    ViT has no attention mask, so it falls through every path except "path5",
    which requires Div instead of Mul. Since Mul(x, s) == Div(x, 1/s), rewriting
    the node in place makes the exported graph match "path5" and fuses correctly
    without changing the model's numerics.

    Returns the number of nodes rewritten (0 means nothing to patch, e.g. if a
    future onnxruntime/transformers release already aligns on this).
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(input_path))
    graph = model.graph

    output_to_node = {out: node for node in graph.node for out in node.output}

    def get_constant_value(name: str):
        for init in graph.initializer:
            if init.name == name:
                return numpy_helper.to_array(init)
        node = output_to_node.get(name)
        if node is not None and node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
        return None

    fixed = 0
    for node in [n for n in graph.node if n.op_type == "Softmax"]:
        parent = output_to_node.get(node.input[0])
        if parent is None or parent.op_type != "Mul":
            continue

        const_idx, const_val = None, None
        for i, inp in enumerate(parent.input):
            val = get_constant_value(inp)
            if val is not None:
                const_idx, const_val = i, val

        if const_idx is None:
            continue  # scale isn't a simple constant; leave it alone

        x_input = parent.input[1 - const_idx]
        new_val = (1.0 / const_val).astype(const_val.dtype)
        new_init_name = parent.input[const_idx] + "_reciprocal"
        graph.initializer.append(numpy_helper.from_array(new_val, name=new_init_name))

        parent.op_type = "Div"
        parent.input[0] = x_input
        parent.input[1] = new_init_name
        fixed += 1

    if fixed:
        onnx.checker.check_model(model)
        onnx.save(model, str(output_path))
    return fixed


def run_attention_fusion(
    input_path: Path,
    output_path: Path,
    num_heads: int,
    hidden_size: int,
) -> None:
    """Run onnxruntime's transformer attention-fusion pass on the exported model.

    opt_level=0 disables onnxruntime's own graph optimizations and only runs the
    python fusion logic, which makes it easy to see exactly which fusions matched
    (via verbose=True) instead of silently falling back to an unfused graph.
    """
    from onnxruntime.transformers.optimizer import optimize_model

    patched_path = input_path.with_name(f"{input_path.stem}_prefused.onnx")
    n_patched = patch_scale_mul_to_div(input_path, patched_path)
    fusion_input = patched_path if n_patched else input_path
    if n_patched:
        print(f"Rewired {n_patched} Mul->Div scaling node(s) before Softmax.")

    optimized = optimize_model(
        str(fusion_input),
        model_type="vit",
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=0,
        verbose=True,
    )

    stats = optimized.get_operator_statistics()
    fused_attention_count = stats.get("Attention", 0)
    if fused_attention_count == 0:
        print(
            "\n[WARNING] No 'Attention' nodes were fused. This almost always means "
            "the exported graph doesn't match the expected pattern (e.g. it still "
            "uses SDPA, or num_heads/hidden_size are wrong). Inspect the model in "
            "Netron and re-check --attn-implementation.",
            file=sys.stderr,
        )
    else:
        print(f"\nFused {fused_attention_count} Attention node(s).")

    optimized.save_model_to_file(str(output_path))


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.optimize == "O4" and not args.device.startswith("cuda"):
        raise ValueError("Optimum O4 requires --device cuda")

    config, height, width, num_channels = resolve_image_configuration(
        model_name=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        height=args.height,
        width=args.width,
        num_channels=args.num_channels,
    )

    if height <= 0 or width <= 0 or num_channels <= 0:
        raise ValueError("Image dimensions and channel count must be positive")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{sanitize_model_name(args.model)}"
        f"_b{args.batch_size}"
        f"_h{height}"
        f"_w{width}"
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

    command = [
        resolve_optimum_cli(),
        "export",
        "onnx",
        "--model",
        args.model,
        "--task",
        args.task,
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
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_channels",
        str(num_channels),
        # Force the classic (unfused) attention graph. onnxruntime's transformer
        # optimizer pattern-matches on this; models traced with SDPA generally
        # fail attention fusion (see microsoft/onnxruntime#21208 for the same
        # issue on CLIP-ViT).
        "--model-kwargs",
        json.dumps({"attn_implementation": args.attn_implementation}),
    ]

    if args.static:
        command.append("--no-dynamic-axes")
    if args.optimize is not None:
        command.extend(["--optimize", args.optimize])
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
    # finalize_model(
    #     model_path,
    #     model_path.with_name(f"{model_path.stem}_optimized.onnx"),
    #     opset=args.opset,
    #     precision=args.precision,
    #     is_transformer=True,
    # )

    num_heads = int(config.num_attention_heads)
    hidden_size = int(config.hidden_size)

    print(f"\nExported model: {model_path}")
    print("Dynamic axes:", "disabled" if args.static else "enabled by Optimum")
    print("Attention implementation forced to:", args.attn_implementation)

    if args.skip_attention_fusion:
        print(
            "\nSkipping attention fusion (--skip-attention-fusion). To run it "
            "manually:\n"
            f"{sys.executable} -m onnxruntime.transformers.optimizer "
            f'--input "{model_path}" '
            f'--output "{model_path.with_name(f"{model_path.stem}_transformer_optimized.onnx")}" '
            f"--model_type vit --num_heads {num_heads} --hidden_size {hidden_size} "
            f"--opt_level 0 --verbose"
        )
        return

    transformer_optimized_path = model_path.with_name(
        f"{model_path.stem}_transformer_optimized.onnx"
    )
    run_attention_fusion(model_path, transformer_optimized_path, num_heads, hidden_size)
    print(f"Transformer-optimized model: {transformer_optimized_path}")


if __name__ == "__main__":
    main()
