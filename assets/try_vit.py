from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

# =============================================================================
# Configuration
# =============================================================================
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
from transformers import ViTForImageClassification

TRANSFORMER_OPTIMIZED_PATH = Path("vit_transformer_optimized.onnx")
MODEL_ID = "google/vit-base-patch16-224"

RAW_PATH = Path("vit_raw.onnx")
OPTIMIZED_PATH = Path("vit_ort_extended.onnx")

BATCH_SIZE = 2
IMAGE_SIZE = 224

OPSET = 24


# =============================================================================
# Wrapper
# =============================================================================


class ViTLogits(torch.nn.Module):
    def __init__(
        self,
        model: ViTForImageClassification,
    ) -> None:
        super().__init__()

        self.model = model

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            pixel_values=pixel_values,
            return_dict=True,
        ).logits


# =============================================================================
# Inspection
# =============================================================================


def print_statistics(
    path: Path,
) -> None:
    model = onnx.load(
        path,
        load_external_data=False,
    )

    counts = Counter(
        (
            node.domain or "ai.onnx",
            node.op_type,
        )
        for node in model.graph.node
    )

    print()
    print("=" * 100)
    print(path)
    print("=" * 100)

    print(
        "Total nodes:",
        len(model.graph.node),
    )

    print()

    for (
        domain,
        op_type,
    ), count in sorted(counts.items()):
        print(f"{domain:30}{op_type:45}{count}")


def count_standard_attention(
    path: Path,
) -> int:
    model = onnx.load(
        path,
        load_external_data=False,
    )

    return sum(
        1
        for node in model.graph.node
        if (node.op_type == "Attention" and node.domain in {"", "ai.onnx"})
    )


def count_microsoft_attention(
    path: Path,
) -> int:
    model = onnx.load(
        path,
        load_external_data=False,
    )

    attention_ops = {
        "Attention",
        "MultiHeadAttention",
        "GroupQueryAttention",
    }

    return sum(
        1
        for node in model.graph.node
        if (node.domain == "com.microsoft" and node.op_type in attention_ops)
    )


# =============================================================================
# Numerical verification
# =============================================================================


def create_session(
    path: Path,
) -> ort.InferenceSession:
    options = ort.SessionOptions()

    # Important:
    # do not optimize again while verifying.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[
            "CPUExecutionProvider",
        ],
    )


def verify_models() -> None:
    print()
    print("=" * 100)
    print("RAW vs ORT EXTENDED")
    print("=" * 100)

    raw_session = create_session(RAW_PATH)

    optimized_session = create_session(TRANSFORMER_OPTIMIZED_PATH)

    np.random.seed(42)

    for batch_size in [
        1,
        2,
        4,
    ]:
        pixel_values = np.random.randn(
            batch_size,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ).astype(np.float32)

        feeds = {
            "pixel_values": pixel_values,
        }

        raw_output = raw_session.run(
            None,
            feeds,
        )[0]

        optimized_output = optimized_session.run(
            None,
            feeds,
        )[0]

        difference = np.abs(raw_output - optimized_output)

        max_error = float(difference.max())

        mean_error = float(difference.mean())

        print()
        print(f"batch={batch_size}")

        print(f"  max error : {max_error:.8e}")

        print(f"  mean error: {mean_error:.8e}")

        np.testing.assert_allclose(
            raw_output,
            optimized_output,
            rtol=1e-4,
            atol=2e-4,
        )


# =============================================================================
# Model
# =============================================================================


torch.manual_seed(42)

model = ViTForImageClassification.from_pretrained(
    MODEL_ID,
    attn_implementation="sdpa",
)

model.eval()

wrapped_model = ViTLogits(model)

wrapped_model.eval()


# =============================================================================
# Example input
# =============================================================================


pixel_values = torch.randn(
    BATCH_SIZE,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
)


batch_dimension = torch.export.Dim(
    "batch_size",
    min=1,
)


# =============================================================================
# Stage 1: PyTorch -> ONNX
# =============================================================================


print()
print("[1] Export")

torch.onnx.export(
    wrapped_model,
    args=(pixel_values,),
    f=RAW_PATH,
    input_names=[
        "pixel_values",
    ],
    output_names=[
        "logits",
    ],
    opset_version=OPSET,
    dynamo=True,
    optimize=True,
    # Only batch is dynamic.
    #
    # Input stays:
    #
    #     [batch_size, 3, 224, 224]
    dynamic_shapes=(
        {
            0: batch_dimension,
        },
    ),
)


raw_model = onnx.load(RAW_PATH)

onnx.checker.check_model(raw_model)


print_statistics(RAW_PATH)


print()
print(
    "Standard Attention:",
    count_standard_attention(RAW_PATH),
)

print(
    "Microsoft Attention:",
    count_microsoft_attention(RAW_PATH),
)


# =============================================================================
# Stage 2: ORT Transformer-specific optimizer
# =============================================================================

print()
print("[2] ONNX Runtime Transformer optimizer")


options = FusionOptions(
    model_type="vit",
)

# Per ora disabiliterei la symbolic shape inference di ORT.
#
# Abbiamo già verificato che ORT 1.27 ha problemi con
# ai.onnx::Attention standard.
#
# Questo ci permette di vedere quali fusioni riconosce
# direttamente dalla struttura del grafo.
options.enable_shape_inference = True


optimized_model = optimizer.optimize_model(
    str(RAW_PATH),
    model_type="vit",
    num_heads=model.config.num_attention_heads,
    hidden_size=model.config.hidden_size,
    # IMPORTANT:
    #
    # 0 => only Transformer-specific Python fusion passes.
    #
    # No BASIC / EXTENDED native ORT optimization is run first.
    opt_level=2,
    optimization_options=options,
)


print()
print("Transformer fusion statistics:")
print()

statistics = optimized_model.get_fused_operator_statistics()

for name, count in statistics.items():
    print(f"  {name:45}{count}")


optimized_model.save_model_to_file(str(TRANSFORMER_OPTIMIZED_PATH))


saved_model = onnx.load(TRANSFORMER_OPTIMIZED_PATH)

onnx.checker.check_model(saved_model)


print_statistics(TRANSFORMER_OPTIMIZED_PATH)


print()
print(
    "Standard Attention:",
    count_standard_attention(TRANSFORMER_OPTIMIZED_PATH),
)

print(
    "Microsoft Attention:",
    count_microsoft_attention(TRANSFORMER_OPTIMIZED_PATH),
)


# =============================================================================
# Comparison
# =============================================================================


raw_nodes = len(raw_model.graph.node)

optimized_nodes = len(optimized_model.graph.node)


print()
print("=" * 100)
print("COMPARISON")
print("=" * 100)

print(f"RAW nodes       : {raw_nodes}")

print(f"EXTENDED nodes  : {optimized_nodes}")

print(
    "Reduction       : "
    f"{raw_nodes - optimized_nodes} "
    f"({100 * (raw_nodes - optimized_nodes) / raw_nodes:.2f}%)"
)


# =============================================================================
# Verification
# =============================================================================


verify_models()


print()
print("=" * 100)
print("DONE")
print("=" * 100)

print(f"RAW       : {RAW_PATH}")

print(f"OPTIMIZED : {OPTIMIZED_PATH}")
