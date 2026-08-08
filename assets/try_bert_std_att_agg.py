from __future__ import annotations

from collections import Counter
from math import sqrt
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxscript
import torch
from onnx import helper
from onnxscript import ir
from onnxscript import optimizer as onnxscript_optimizer
from onnxscript.rewriter import pattern
from transformers import BertModel

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = "bert-base-uncased"

RAW_PATH = Path("bert_raw.onnx")
REWRITTEN_PATH = Path("bert_attention_3d.onnx")

OPSET = 24

EXAMPLE_BATCH_SIZE = 2
EXAMPLE_SEQUENCE_LENGTH = 32

SEED = 42


# =============================================================================
# Wrapper
# =============================================================================


class BertWrapper(torch.nn.Module):
    def __init__(self, model: BertModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        # last_hidden_state:
        #
        # [batch_size, sequence_length, hidden_size]
        return outputs[0]


# =============================================================================
# ONNX utilities
# =============================================================================


def get_attribute(
    node: onnx.NodeProto,
    name: str,
):
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)

    return None


def count_standard_attention(
    model: onnx.ModelProto,
) -> int:
    return sum(
        1
        for node in model.graph.node
        if node.op_type == "Attention" and node.domain in {"", "ai.onnx"}
    )


def count_3d_attention(
    model: onnx.ModelProto,
) -> int:
    """
    After our rewrite, q_num_heads is explicitly present.

    The original 4D Attention does not need q_num_heads.
    """
    return sum(
        1
        for node in model.graph.node
        if (
            node.op_type == "Attention"
            and node.domain in {"", "ai.onnx"}
            and get_attribute(node, "q_num_heads") is not None
        )
    )


def print_statistics(
    title: str,
    model: onnx.ModelProto,
) -> None:
    counts = Counter(
        (
            node.domain or "ai.onnx",
            node.op_type,
        )
        for node in model.graph.node
    )

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print(f"Total nodes          : {len(model.graph.node)}")

    print(f"Standard Attention   : {count_standard_attention(model)}")

    print(f"3D Attention         : {count_3d_attention(model)}")

    print()

    interesting_ops = (
        "Attention",
        "Reshape",
        "Transpose",
        "Shape",
        "Gather",
        "Unsqueeze",
        "Concat",
        "MatMul",
        "Add",
        "LayerNormalization",
        "Gelu",
    )

    for op_type in interesting_ops:
        count = sum(
            number
            for (_, current_op), number in counts.items()
            if current_op == op_type
        )

        print(f"{op_type:25}: {count}")


# =============================================================================
# Attention rewrite
# =============================================================================


def make_attention_with_mask_rule(
    *,
    num_heads: int,
    head_size: int,
) -> pattern.RewriteRule:
    """
    Match:

        Q3 -> Reshape -> Transpose --\
        K3 -> Reshape -> Transpose --- Attention4D -> Transpose -> Reshape
        V3 -> Reshape -> Transpose --/      ^
                                             |
                                          mask

    and replace with:

        Q3 --\
        K3 --- Attention3D
        V3 --/      ^
                    |
                   mask
    """

    def target_pattern(
        op,
        q,
        k,
        v,
        q_shape,
        k_shape,
        v_shape,
        attention_mask,
        output_shape,
    ):
        q_reshaped = op.Reshape(
            q,
            q_shape,
        )

        q_4d = op.Transpose(
            q_reshaped,
            perm=[0, 2, 1, 3],
        )

        k_reshaped = op.Reshape(
            k,
            k_shape,
        )

        k_4d = op.Transpose(
            k_reshaped,
            perm=[0, 2, 1, 3],
        )

        v_reshaped = op.Reshape(
            v,
            v_shape,
        )

        v_4d = op.Transpose(
            v_reshaped,
            perm=[0, 2, 1, 3],
        )

        attention = op.Attention(
            q_4d,
            k_4d,
            v_4d,
            attention_mask,
            _domain="",
            _allow_other_attributes=True,
        )

        attention_transposed = op.Transpose(
            attention,
            perm=[0, 2, 1, 3],
        )

        return op.Reshape(
            attention_transposed,
            output_shape,
        )

    def replacement_pattern(
        op,
        q: ir.Value,
        k: ir.Value,
        v: ir.Value,
        attention_mask: ir.Value,
        **_,
    ):
        return op.Attention(
            q,
            k,
            v,
            attention_mask,
            q_num_heads=num_heads,
            kv_num_heads=num_heads,
            # BERT standard scaled-dot-product attention.
            scale=1.0 / sqrt(head_size),
            is_causal=0,
            _domain="",
        )

    def match_condition(
        context,
        q: ir.Value,
        k: ir.Value,
        v: ir.Value,
        **_,
    ) -> bool:
        del context

        # The whole point of this transformation is to feed the
        # pre-head-splitting [B, S, hidden] tensors directly into
        # standard ONNX Attention.
        for value in (q, k, v):
            if value.shape is None:
                return False

            if len(value.shape) != 3:
                return False

        return True

    return pattern.RewriteRule(
        target_pattern,
        replacement_pattern,
        match_condition,
    )


def make_attention_without_mask_rule(
    *,
    num_heads: int,
    head_size: int,
) -> pattern.RewriteRule:
    """
    Same transformation for Attention nodes without an explicit mask.

    BERT normally uses the masked rule, but keeping this rule makes the
    transformation robust when the exporter removes an unnecessary mask.
    """

    def target_pattern(
        op,
        q,
        k,
        v,
        q_shape,
        k_shape,
        v_shape,
        output_shape,
    ):
        q_reshaped = op.Reshape(
            q,
            q_shape,
        )

        q_4d = op.Transpose(
            q_reshaped,
            perm=[0, 2, 1, 3],
        )

        k_reshaped = op.Reshape(
            k,
            k_shape,
        )

        k_4d = op.Transpose(
            k_reshaped,
            perm=[0, 2, 1, 3],
        )

        v_reshaped = op.Reshape(
            v,
            v_shape,
        )

        v_4d = op.Transpose(
            v_reshaped,
            perm=[0, 2, 1, 3],
        )

        attention = op.Attention(
            q_4d,
            k_4d,
            v_4d,
            _domain="",
            _allow_other_attributes=True,
        )

        attention_transposed = op.Transpose(
            attention,
            perm=[0, 2, 1, 3],
        )

        return op.Reshape(
            attention_transposed,
            output_shape,
        )

    def replacement_pattern(
        op,
        q: ir.Value,
        k: ir.Value,
        v: ir.Value,
        **_,
    ):
        return op.Attention(
            q,
            k,
            v,
            q_num_heads=num_heads,
            kv_num_heads=num_heads,
            scale=1.0 / sqrt(head_size),
            is_causal=0,
            _domain="",
        )

    def match_condition(
        context,
        q: ir.Value,
        k: ir.Value,
        v: ir.Value,
        **_,
    ) -> bool:
        del context

        return all(
            value.shape is not None and len(value.shape) == 3 for value in (q, k, v)
        )

    return pattern.RewriteRule(
        target_pattern,
        replacement_pattern,
        match_condition,
    )


# =============================================================================
# Input generation
# =============================================================================


def make_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    type_vocab_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    input_ids = rng.integers(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            sequence_length,
        ),
        dtype=np.int64,
    )

    token_type_ids = rng.integers(
        low=0,
        high=type_vocab_size,
        size=(
            batch_size,
            sequence_length,
        ),
        dtype=np.int64,
    )

    # Use actual padding instead of an all-one mask so that the
    # Attention-mask path is numerically tested too.
    attention_mask = np.ones(
        (
            batch_size,
            sequence_length,
        ),
        dtype=np.int64,
    )

    if sequence_length >= 4:
        for batch_index in range(batch_size):
            padding = min(
                batch_index + 1,
                sequence_length // 4,
            )

            if padding > 0:
                attention_mask[batch_index, -padding:] = 0

    return (
        input_ids,
        attention_mask,
        token_type_ids,
    )


# =============================================================================
# ORT verification
# =============================================================================


def create_session(
    path: Path,
) -> ort.InferenceSession:
    options = ort.SessionOptions()

    # Important:
    #
    # validate the actual serialized graph rather than comparing two
    # graphs after ORT has aggressively optimized them again.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[
            "CPUExecutionProvider",
        ],
    )


def verify_models(
    *,
    raw_path: Path,
    rewritten_path: Path,
    vocab_size: int,
    type_vocab_size: int,
) -> None:
    raw_session = create_session(raw_path)

    rewritten_session = create_session(rewritten_path)

    tests = (
        (1, 8),
        (2, 17),
        (4, 32),
        (1, 64),
    )

    print()
    print("=" * 100)
    print("NUMERICAL VERIFICATION")
    print("=" * 100)

    for test_index, (
        batch_size,
        sequence_length,
    ) in enumerate(tests):
        (
            input_ids,
            attention_mask,
            token_type_ids,
        ) = make_inputs(
            batch_size=batch_size,
            sequence_length=sequence_length,
            vocab_size=vocab_size,
            type_vocab_size=type_vocab_size,
            seed=SEED + test_index,
        )

        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        raw_output = raw_session.run(
            None,
            feeds,
        )[0]

        rewritten_output = rewritten_session.run(
            None,
            feeds,
        )[0]

        difference = np.abs(raw_output - rewritten_output)

        max_error = float(difference.max())

        mean_error = float(difference.mean())

        print(
            f"batch={batch_size:2}, "
            f"seq={sequence_length:3} | "
            f"max={max_error:.8e} | "
            f"mean={mean_error:.8e}"
        )

        np.testing.assert_allclose(
            raw_output,
            rewritten_output,
            rtol=1e-4,
            atol=2e-4,
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    torch.manual_seed(SEED)

    # -------------------------------------------------------------------------
    # Load BERT
    # -------------------------------------------------------------------------

    print("[1] Loading BERT")

    model = BertModel.from_pretrained(
        MODEL_ID,
        # We only need last_hidden_state.
        add_pooling_layer=False,
        # Encourage export through scaled-dot-product attention.
        attn_implementation="sdpa",
    )

    model.eval()

    wrapped_model = BertWrapper(model)

    wrapped_model.eval()

    config = model.config

    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size

    assert hidden_size % num_heads == 0

    head_size = hidden_size // num_heads

    print(f"layers      : {config.num_hidden_layers}")

    print(f"hidden size : {hidden_size}")

    print(f"heads       : {num_heads}")

    print(f"head size   : {head_size}")

    # -------------------------------------------------------------------------
    # Example export inputs
    # -------------------------------------------------------------------------

    (
        example_input_ids_np,
        example_attention_mask_np,
        example_token_type_ids_np,
    ) = make_inputs(
        batch_size=EXAMPLE_BATCH_SIZE,
        sequence_length=EXAMPLE_SEQUENCE_LENGTH,
        vocab_size=config.vocab_size,
        type_vocab_size=config.type_vocab_size,
        seed=SEED,
    )

    example_input_ids = torch.from_numpy(example_input_ids_np)

    example_attention_mask = torch.from_numpy(example_attention_mask_np)

    example_token_type_ids = torch.from_numpy(example_token_type_ids_np)

    # -------------------------------------------------------------------------
    # Dynamic dimensions
    # -------------------------------------------------------------------------

    batch_dimension = torch.export.Dim(
        "batch_size",
        min=1,
    )

    sequence_dimension = torch.export.Dim(
        "sequence_length",
        min=1,
        max=config.max_position_embeddings,
    )

    dynamic_shapes = (
        {
            0: batch_dimension,
            1: sequence_dimension,
        },
        {
            0: batch_dimension,
            1: sequence_dimension,
        },
        {
            0: batch_dimension,
            1: sequence_dimension,
        },
    )

    # -------------------------------------------------------------------------
    # PyTorch -> ONNX
    # -------------------------------------------------------------------------

    print()
    print("[2] Exporting RAW ONNX")

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            args=(
                example_input_ids,
                example_attention_mask,
                example_token_type_ids,
            ),
            f=RAW_PATH,
            input_names=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
            ],
            output_names=[
                "last_hidden_state",
            ],
            opset_version=OPSET,
            dynamo=True,
            # Let the modern exporter perform its normal cleanups.
            optimize=True,
            dynamic_shapes=dynamic_shapes,
            # BERT-base is below the protobuf 2GB limit, so using a
            # single self-contained ONNX file simplifies rewriting.
            external_data=False,
        )

    raw_model = onnx.load(RAW_PATH)

    onnx.checker.check_model(raw_model)

    print_statistics(
        "RAW ONNX",
        raw_model,
    )

    expected_attention_count = config.num_hidden_layers

    raw_attention_count = count_standard_attention(raw_model)

    if raw_attention_count != expected_attention_count:
        raise RuntimeError(
            "Unexpected number of standard Attention nodes: "
            f"expected {expected_attention_count}, "
            f"found {raw_attention_count}"
        )

    # -------------------------------------------------------------------------
    # Standard ONNX shape inference
    # -------------------------------------------------------------------------

    print()
    print("[3] ONNX shape inference")

    # This is ONNX shape inference, NOT ORT symbolic shape inference.
    #
    # It helps populate rank information consumed by the rewrite condition.
    raw_model = onnx.shape_inference.infer_shapes(raw_model)

    # -------------------------------------------------------------------------
    # ONNX Script rewrite
    # -------------------------------------------------------------------------

    print()
    print("[4] Attention 4D -> 3D rewrite")

    masked_rule = make_attention_with_mask_rule(
        num_heads=num_heads,
        head_size=head_size,
    )

    unmasked_rule = make_attention_without_mask_rule(
        num_heads=num_heads,
        head_size=head_size,
    )

    rewritten_model = onnxscript.rewriter.rewrite(
        raw_model,
        pattern_rewrite_rules=[
            # Try the BERT case first.
            masked_rule,
            unmasked_rule,
        ],
    )

    print_statistics(
        "AFTER ATTENTION REWRITE",
        rewritten_model,
    )

    rewritten_attention_count = count_3d_attention(rewritten_model)

    if rewritten_attention_count != expected_attention_count:
        raise RuntimeError(
            "Attention rewrite did not match every BERT block: "
            f"expected {expected_attention_count}, "
            f"rewritten {rewritten_attention_count}"
        )

    # -------------------------------------------------------------------------
    # Dead-code elimination / constant folding
    # -------------------------------------------------------------------------

    print()
    print("[5] ONNX Script cleanup")

    rewritten_model = onnxscript_optimizer.optimize(
        rewritten_model,
        # A couple of passes are normally enough for shape machinery.
        num_iterations=2,
        onnx_shape_inference=True,
        # Critical: ai.onnx::Attention is itself an ONNX Function.
        # Keep it as a semantic Attention node.
        inline=False,
    )

    onnx.checker.check_model(rewritten_model)

    print_statistics(
        "AFTER CLEANUP",
        rewritten_model,
    )

    final_attention_count = count_3d_attention(rewritten_model)

    if final_attention_count != expected_attention_count:
        raise RuntimeError(
            "Attention nodes were unexpectedly changed during cleanup: "
            f"expected {expected_attention_count}, "
            f"found {final_attention_count}"
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    print()
    print("[6] Saving")

    onnx.save(
        rewritten_model,
        REWRITTEN_PATH,
    )

    # Reload from disk to test exactly what was serialized.
    final_model = onnx.load(REWRITTEN_PATH)

    onnx.checker.check_model(final_model)

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------

    raw_model_disk = onnx.load(RAW_PATH)

    raw_nodes = len(raw_model_disk.graph.node)

    rewritten_nodes = len(final_model.graph.node)

    reduction = raw_nodes - rewritten_nodes

    reduction_percentage = 100.0 * reduction / raw_nodes

    print()
    print("=" * 100)
    print("GRAPH COMPARISON")
    print("=" * 100)

    print(f"RAW nodes          : {raw_nodes}")

    print(f"Rewritten nodes    : {rewritten_nodes}")

    print(f"Removed nodes      : {reduction}")

    print(f"Reduction          : {reduction_percentage:.2f}%")

    # -------------------------------------------------------------------------
    # Numerical verification
    # -------------------------------------------------------------------------

    print()
    print("[7] Numerical verification")

    verify_models(
        raw_path=RAW_PATH,
        rewritten_path=REWRITTEN_PATH,
        vocab_size=config.vocab_size,
        type_vocab_size=config.type_vocab_size,
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(f"RAW       : {RAW_PATH}")

    print(f"REWRITTEN : {REWRITTEN_PATH}")


if __name__ == "__main__":
    main()
