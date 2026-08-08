from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import onnxscript
import torch
from onnx import helper
from onnxscript import ir
from onnxscript import optimizer as onnxscript_optimizer
from onnxscript.rewriter import pattern
from transformers import DynamicCache, GPT2LMHeadModel

# =============================================================================
# Configuration
# =============================================================================

MODEL_ID = "openai-community/gpt2"

RAW_PATH = Path("gpt2_raw.onnx")
ATTENTION_PATH = Path("gpt2_attention_rewritten.onnx")
FINAL_PATH = Path("gpt2_final.onnx")

OPSET = 24

EXAMPLE_BATCH_SIZE = 2
EXAMPLE_CURRENT_LENGTH = 4
EXAMPLE_PAST_LENGTH = 8

SEED = 42


# =============================================================================
# PyTorch wrapper
# =============================================================================


class GPT2WithExplicitCache(torch.nn.Module):
    """
    ONNX-visible interface:

        input_ids

        past_key_0
        past_value_0
        ...
        past_key_N
        past_value_N

            ↓

        logits

        present_key_0
        present_value_0
        ...
        present_key_N
        present_value_N

    The Hugging Face DynamicCache exists only inside this adapter.
    """

    def __init__(
        self,
        model: GPT2LMHeadModel,
    ) -> None:
        super().__init__()

        self.model = model

        self.num_layers = model.config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        flat_past: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:

        # ---------------------------------------------------------------------
        # Explicit tensor cache -> DynamicCache
        # ---------------------------------------------------------------------

        cache_data = tuple(
            (
                flat_past[2 * layer_index],
                flat_past[2 * layer_index + 1],
            )
            for layer_index in range(self.num_layers)
        )

        cache = DynamicCache(
            ddp_cache_data=cache_data,
            config=self.model.config,
        )

        # ---------------------------------------------------------------------
        # GPT-2
        # ---------------------------------------------------------------------

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        assert outputs.past_key_values is not None

        # ---------------------------------------------------------------------
        # DynamicCache -> explicit tensors
        # ---------------------------------------------------------------------

        legacy_cache = outputs.past_key_values.to_legacy_cache()

        flat_present = tuple(
            tensor for layer_cache in legacy_cache for tensor in layer_cache
        )

        return (
            outputs.logits,
            *flat_present,
        )


# =============================================================================
# ONNX helpers
# =============================================================================


def make_gpt2_gelu_new_rule() -> pattern.RewriteRule:
    """
    Hugging Face GPT-2 NewGELU:

        0.5 * x * (
            1.0
            + tanh(
                sqrt(2/pi)
                * (
                    x
                    + 0.044715 * x**3
                )
            )
        )
    """

    def target_pattern(
        op,
        x,
    ):
        x_cubed = op.Pow(
            x,
            3.0,
        )

        cubic_term = op.Mul(
            x_cubed,
            0.044715,
        )

        inner = op.Add(
            x,
            cubic_term,
        )

        scaled = op.Mul(
            inner,
            math.sqrt(2.0 / math.pi),
        )

        tanh = op.Tanh(
            scaled,
        )

        one_plus_tanh = op.Add(
            tanh,
            1.0,
        )

        half_x = op.Mul(
            x,
            0.5,
        )

        return op.Mul(
            half_x,
            one_plus_tanh,
        )

    def replacement_pattern(
        op,
        x: ir.Value,
        **_,
    ):
        return op.Gelu(
            x,
            approximate="tanh",
            _domain="",
        )

    return pattern.RewriteRule(
        target_pattern,
        replacement_pattern,
    )


def get_attribute(
    node: onnx.NodeProto,
    name: str,
    default: Any = None,
) -> Any:

    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)

    return default


def is_standard_attention(
    node: onnx.NodeProto,
) -> bool:

    return node.op_type == "Attention" and node.domain in {
        "",
        "ai.onnx",
    }


def count_attention(
    model: onnx.ModelProto,
) -> int:

    return sum(is_standard_attention(node) for node in model.graph.node)


def count_3d_cache_attention(
    model: onnx.ModelProto,
) -> int:
    """
    Identify our final Attention nodes.

    They must have:
        q_num_heads
        kv_num_heads
        past_key
        past_value
        present_key
        present_value
    """

    count = 0

    for node in model.graph.node:
        if not is_standard_attention(node):
            continue

        if (
            get_attribute(
                node,
                "q_num_heads",
            )
            is None
        ):
            continue

        # Q K V mask past_key past_value
        if len(node.input) < 6:
            continue

        # Y present_key present_value
        if len(node.output) < 3:
            continue

        count += 1

    return count


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

    print(f"Total nodes              : {len(model.graph.node)}")

    print(f"Standard Attention       : {count_attention(model)}")

    print(f"3D Attention + KV cache  : {count_3d_cache_attention(model)}")

    print()

    interesting_ops = (
        "Attention",
        "Gemm",
        "MatMul",
        "Add",
        "Reshape",
        "Transpose",
        "Concat",
        "Split",
        "Shape",
        "Gather",
        "Unsqueeze",
        "Squeeze",
        "Expand",
        "Range",
        "Where",
        "LayerNormalization",
    )

    for op_type in interesting_ops:
        count = sum(
            number
            for (_, current_op), number in counts.items()
            if current_op == op_type
        )

        print(f"{op_type:28}: {count}")


# =============================================================================
# Validate Gemm semantics
# =============================================================================


def validate_gpt2_gemms(
    model: onnx.ModelProto,
) -> None:
    """
    Our Gemm -> MatMul + Add rewrite assumes:

        Y = A @ B + C

    i.e.

        alpha  = 1
        beta   = 1
        transA = 0
        transB = 0

    These are exactly the semantics of Hugging Face GPT-2 Conv1D's
    addmm-based implementation.

    Abort rather than silently rewrite something with different semantics.
    """

    for node in model.graph.node:
        if node.op_type != "Gemm":
            continue

        alpha = float(
            get_attribute(
                node,
                "alpha",
                1.0,
            )
        )

        beta = float(
            get_attribute(
                node,
                "beta",
                1.0,
            )
        )

        trans_a = int(
            get_attribute(
                node,
                "transA",
                0,
            )
        )

        trans_b = int(
            get_attribute(
                node,
                "transB",
                0,
            )
        )

        if not (alpha == 1.0 and beta == 1.0 and trans_a == 0 and trans_b == 0):
            raise RuntimeError(
                "Unsupported GPT-2 Gemm semantics:\n"
                f"  node   = {node.name}\n"
                f"  alpha  = {alpha}\n"
                f"  beta   = {beta}\n"
                f"  transA = {trans_a}\n"
                f"  transB = {trans_b}"
            )


# =============================================================================
# Rewrite #1:
#
# 4D Attention + external KV Concat
#
#       ↓
#
# 3D Attention with internal KV cache
# =============================================================================


def make_gpt2_cache_attention_rule(
    *,
    num_heads: int,
    cache_axis: int,
    masked: bool,
) -> pattern.RewriteRule:

    # =========================================================================
    # Variant with explicit attention mask
    # =========================================================================

    if masked:

        def target_pattern(
            op,
            q,
            k,
            v,
            q_shape,
            k_shape,
            v_shape,
            past_key,
            past_value,
            attention_mask,
        ):
            # -------------------------------------------------------------
            # Q: [B,S,H] -> [B,heads,S,D]
            # -------------------------------------------------------------

            q_4d = op.Transpose(
                op.Reshape(
                    q,
                    q_shape,
                    _allow_other_attributes=True,
                ),
                perm=[0, 2, 1, 3],
            )

            # -------------------------------------------------------------
            # Current K
            # -------------------------------------------------------------

            current_key = op.Transpose(
                op.Reshape(
                    k,
                    k_shape,
                    _allow_other_attributes=True,
                ),
                perm=[0, 2, 1, 3],
            )

            present_key = op.Concat(
                past_key,
                current_key,
                axis=cache_axis,
            )

            # -------------------------------------------------------------
            # Current V
            # -------------------------------------------------------------

            current_value = op.Transpose(
                op.Reshape(
                    v,
                    v_shape,
                    _allow_other_attributes=True,
                ),
                perm=[0, 2, 1, 3],
            )

            present_value = op.Concat(
                past_value,
                current_value,
                axis=cache_axis,
            )

            # -------------------------------------------------------------
            # Original 4D Attention
            # -------------------------------------------------------------

            attention = op.Attention(
                q_4d,
                present_key,
                present_value,
                attention_mask,
                _domain="",
                _allow_other_attributes=True,
            )

            # [B,H,S,D] -> [B,S,H,D]
            attention_transposed = op.Transpose(
                attention,
                perm=[0, 2, 1, 3],
            )

            # present K/V are part of model outputs.
            return (
                attention_transposed,
                present_key,
                present_value,
            )

        def replacement_pattern(
            op,
            q: ir.Value,
            k: ir.Value,
            v: ir.Value,
            past_key: ir.Value,
            past_value: ir.Value,
            attention_mask: ir.Value,
            **_,
        ):
            # attention_mask intentionally ignored.
            #
            # This GPT-2 export has no external padding mask:
            # the mask generated by Transformers is purely causal.
            del attention_mask

            (
                y,
                present_key,
                present_value,
            ) = op.Attention(
                q,
                k,
                v,
                # No explicit attention mask.
                None,
                past_key,
                past_value,
                q_num_heads=num_heads,
                kv_num_heads=num_heads,
                # Let standard ONNX Attention generate the causal mask.
                is_causal=1,
                _domain="",
                _outputs=3,
            )

            return (
                y,
                present_key,
                present_value,
            )

    # =========================================================================
    # Variant without explicit attention mask
    # =========================================================================

    else:

        def target_pattern(
            op,
            q,
            k,
            v,
            q_shape,
            k_shape,
            v_shape,
            past_key,
            past_value,
        ):
            q_4d = op.Transpose(
                op.Reshape(
                    q,
                    q_shape,
                    _allow_other_attributes=True,
                ),
                perm=[0, 2, 1, 3],
            )

            current_key = op.Transpose(
                op.Reshape(
                    k,
                    k_shape,
                    _allow_other_attributes=True,
                ),
                perm=[0, 2, 1, 3],
            )

            present_key = op.Concat(
                past_key,
                current_key,
                axis=cache_axis,
            )

            current_value = op.Transpose(
                op.Reshape(
                    v,
                    v_shape,
                    _allow_other_attributes=True,
                ),
                perm=[0, 2, 1, 3],
            )

            present_value = op.Concat(
                past_value,
                current_value,
                axis=cache_axis,
            )

            attention = op.Attention(
                q_4d,
                present_key,
                present_value,
                _domain="",
                _allow_other_attributes=True,
            )

            attention_transposed = op.Transpose(
                attention,
                perm=[0, 2, 1, 3],
            )

            return (
                attention_transposed,
                present_key,
                present_value,
            )

        def replacement_pattern(
            op,
            q: ir.Value,
            k: ir.Value,
            v: ir.Value,
            past_key: ir.Value,
            past_value: ir.Value,
            **_,
        ):
            (
                y,
                present_key,
                present_value,
            ) = op.Attention(
                q,
                k,
                v,
                # Optional attention-mask input.
                None,
                past_key,
                past_value,
                q_num_heads=num_heads,
                kv_num_heads=num_heads,
                # Causal mask is now performed by Attention itself.
                is_causal=1,
                _domain="",
                _outputs=3,
            )

            return (
                y,
                present_key,
                present_value,
            )

    # =========================================================================
    # Safety condition
    # =========================================================================

    def match_condition(
        context,
        q: ir.Value,
        k: ir.Value,
        v: ir.Value,
        past_key: ir.Value,
        past_value: ir.Value,
        **_,
    ) -> bool:

        del context

        # Q/K/V before head splitting:
        #
        # [B,S,H]
        for value in (
            q,
            k,
            v,
        ):
            if value.shape is None:
                return False

            if len(value.shape) != 3:
                return False

        # Cache:
        #
        # [B,heads,P,D]
        for value in (
            past_key,
            past_value,
        ):
            if value.shape is None:
                return False

            if len(value.shape) != 4:
                return False

        return True

    return pattern.RewriteRule(
        target_pattern,
        replacement_pattern,
        match_condition,
    )


# =============================================================================
# Rewrite #2:
#
# GPT-2 Conv1D
#
# Reshape -> Gemm -> Reshape
#
#          ↓
#
# MatMul -> Add
# =============================================================================


def make_gpt2_conv1d_rule() -> pattern.RewriteRule:
    """
    Match GPT-2 Conv1D export:

        x [B,S,I]
             |
          Reshape
             |
         [B*S,I]
             |
           Gemm
             |
         [B*S,O]
             |
          Reshape
             |
         [B,S,O]

    Replace with:

        x [B,S,I]
             |
          MatMul
             |
            Add
             |
         [B,S,O]


    MatMul supports N-D inputs and broadcasts over B,S.
    Bias Add broadcasts over the final O dimension.
    """

    def target_pattern(
        op,
        x,
        input_shape,
        weight,
        bias,
        output_shape,
    ):
        flattened = op.Reshape(
            x,
            input_shape,
            _allow_other_attributes=True,
        )

        projected_2d = op.Gemm(
            flattened,
            weight,
            bias,
            # Attribute semantics were validated globally before this rule.
            _allow_other_attributes=True,
        )

        projected = op.Reshape(
            projected_2d,
            output_shape,
            _allow_other_attributes=True,
        )

        return projected

    def replacement_pattern(
        op,
        x: ir.Value,
        weight: ir.Value,
        bias: ir.Value,
        **_,
    ):
        projected = op.MatMul(
            x,
            weight,
        )

        return op.Add(
            projected,
            bias,
        )

    def match_condition(
        context,
        x: ir.Value,
        weight: ir.Value,
        bias: ir.Value,
        **_,
    ) -> bool:

        del context

        # GPT-2 Conv1D input:
        #
        # [B,S,input_features]
        if x.shape is None or len(x.shape) != 3:
            return False

        # Weight:
        #
        # [input_features, output_features]
        if weight.shape is None or len(weight.shape) != 2:
            return False

        # Bias:
        #
        # [output_features]
        if bias.shape is None or len(bias.shape) != 1:
            return False

        return True

    return pattern.RewriteRule(
        target_pattern,
        replacement_pattern,
        match_condition,
    )


# =============================================================================
# Test data
# =============================================================================


def make_inputs(
    *,
    batch_size: int,
    current_length: int,
    past_length: int,
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    seed: int,
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, ...],
]:

    rng = np.random.default_rng(seed)

    input_ids = rng.integers(
        0,
        vocab_size,
        size=(
            batch_size,
            current_length,
        ),
        dtype=np.int64,
    )

    flat_past: list[np.ndarray] = []

    for _ in range(num_layers):
        key = rng.standard_normal(
            (
                batch_size,
                num_heads,
                past_length,
                head_size,
            )
        ).astype(np.float32)

        value = rng.standard_normal(
            (
                batch_size,
                num_heads,
                past_length,
                head_size,
            )
        ).astype(np.float32)

        flat_past.extend(
            (
                key,
                value,
            )
        )

    return (
        input_ids,
        tuple(flat_past),
    )


# =============================================================================
# ORT
# =============================================================================


def create_session(
    path: Path,
) -> ort.InferenceSession:

    options = ort.SessionOptions()

    # Compare exactly our serialized graphs.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[
            "CPUExecutionProvider",
        ],
    )


def create_feed(
    *,
    input_ids: np.ndarray,
    flat_past: tuple[np.ndarray, ...],
    num_layers: int,
) -> dict[str, np.ndarray]:

    feed = {
        "input_ids": input_ids,
    }

    for layer_index in range(num_layers):
        feed[f"past_key_{layer_index}"] = flat_past[2 * layer_index]

        feed[f"past_value_{layer_index}"] = flat_past[2 * layer_index + 1]

    return feed


# =============================================================================
# Verification
# =============================================================================


def verify_models(
    *,
    raw_path: Path,
    final_path: Path,
    config,
) -> None:

    raw_session = create_session(raw_path)

    final_session = create_session(final_path)

    num_layers = config.num_hidden_layers

    num_heads = config.num_attention_heads

    head_size = config.hidden_size // num_heads

    tests = (
        # Normal decode.
        (1, 1, 5),
        # Batched + multiple current tokens.
        (2, 3, 7),
        # Longer cache.
        (1, 4, 16),
        # Different dynamic batch/cache dimensions.
        (3, 2, 11),
    )

    print()
    print("=" * 100)
    print("RAW vs FINAL")
    print("=" * 100)

    for test_index, (
        batch_size,
        current_length,
        past_length,
    ) in enumerate(tests):
        (
            input_ids,
            flat_past,
        ) = make_inputs(
            batch_size=batch_size,
            current_length=current_length,
            past_length=past_length,
            vocab_size=config.vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_size=head_size,
            seed=SEED + 100 + test_index,
        )

        feed = create_feed(
            input_ids=input_ids,
            flat_past=flat_past,
            num_layers=num_layers,
        )

        raw_outputs = raw_session.run(
            None,
            feed,
        )

        final_outputs = final_session.run(
            None,
            feed,
        )

        if len(raw_outputs) != len(final_outputs):
            raise RuntimeError("Output count changed.")

        # ---------------------------------------------------------------------
        # Logits
        # ---------------------------------------------------------------------

        logits_diff = np.abs(raw_outputs[0] - final_outputs[0])

        np.testing.assert_allclose(
            raw_outputs[0],
            final_outputs[0],
            rtol=1e-4,
            atol=3e-4,
        )

        # ---------------------------------------------------------------------
        # KV cache
        # ---------------------------------------------------------------------

        max_cache_error = 0.0

        for (
            raw_cache,
            final_cache,
        ) in zip(
            raw_outputs[1:],
            final_outputs[1:],
            strict=True,
        ):
            difference = np.abs(raw_cache - final_cache)

            max_cache_error = max(
                max_cache_error,
                float(difference.max()),
            )

            np.testing.assert_allclose(
                raw_cache,
                final_cache,
                rtol=1e-4,
                atol=3e-4,
            )

        print(
            f"B={batch_size:2} "
            f"S={current_length:3} "
            f"P={past_length:3} | "
            f"logits max={logits_diff.max():.8e} | "
            f"cache max={max_cache_error:.8e}"
        )


# =============================================================================
# Cache round-trip verification
# =============================================================================


def verify_cache_round_trip(
    *,
    raw_path: Path,
    final_path: Path,
    config,
) -> None:
    """
    Perform two consecutive decoding invocations.

    present KV from invocation #1 is fed as past KV of invocation #2.

    This verifies the model interface we actually want to use at runtime.
    """

    raw_session = create_session(raw_path)

    final_session = create_session(final_path)

    num_layers = config.num_hidden_layers

    num_heads = config.num_attention_heads

    head_size = config.hidden_size // num_heads

    (
        first_ids,
        initial_cache,
    ) = make_inputs(
        batch_size=1,
        current_length=1,
        past_length=7,
        vocab_size=config.vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        seed=SEED + 1000,
    )

    # -------------------------------------------------------------------------
    # Step 1
    # -------------------------------------------------------------------------

    raw_step_1 = raw_session.run(
        None,
        create_feed(
            input_ids=first_ids,
            flat_past=initial_cache,
            num_layers=num_layers,
        ),
    )

    final_step_1 = final_session.run(
        None,
        create_feed(
            input_ids=first_ids,
            flat_past=initial_cache,
            num_layers=num_layers,
        ),
    )

    # -------------------------------------------------------------------------
    # Step 2:
    #
    # feed previous present KV back as past KV.
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(SEED + 2000)

    second_ids = rng.integers(
        0,
        config.vocab_size,
        size=(
            1,
            1,
        ),
        dtype=np.int64,
    )

    raw_cache = tuple(raw_step_1[1:])

    final_cache = tuple(final_step_1[1:])

    raw_step_2 = raw_session.run(
        None,
        create_feed(
            input_ids=second_ids,
            flat_past=raw_cache,
            num_layers=num_layers,
        ),
    )

    final_step_2 = final_session.run(
        None,
        create_feed(
            input_ids=second_ids,
            flat_past=final_cache,
            num_layers=num_layers,
        ),
    )

    difference = np.abs(raw_step_2[0] - final_step_2[0])

    np.testing.assert_allclose(
        raw_step_2[0],
        final_step_2[0],
        rtol=1e-4,
        atol=3e-4,
    )

    print()
    print("=" * 100)
    print("CACHE ROUND-TRIP")
    print("=" * 100)

    print(f"Second-step logits max error: {difference.max():.8e}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    torch.manual_seed(SEED)

    # =========================================================================
    # 1. Load model
    # =========================================================================

    print("[1] Loading GPT-2")

    model = GPT2LMHeadModel.from_pretrained(
        MODEL_ID,
        # We want SDPA to be represented semantically
        # by modern PyTorch/ONNX export.
        attn_implementation="sdpa",
    )

    model.eval()

    model.config.use_cache = True

    config = model.config

    num_layers = config.num_hidden_layers

    num_heads = config.num_attention_heads

    hidden_size = config.hidden_size

    assert hidden_size % num_heads == 0

    head_size = hidden_size // num_heads

    # Standard GPT-2 uses normal 1/sqrt(head_size)
    # attention scaling.
    assert config.scale_attn_weights

    assert not (config.scale_attn_by_inverse_layer_idx)

    print(f"layers      : {num_layers}")

    print(f"hidden size : {hidden_size}")

    print(f"heads       : {num_heads}")

    print(f"head size   : {head_size}")

    wrapped_model = GPT2WithExplicitCache(model)

    wrapped_model.eval()

    # =========================================================================
    # 2. Example inputs
    # =========================================================================

    (
        example_input_ids_np,
        example_flat_past_np,
    ) = make_inputs(
        batch_size=EXAMPLE_BATCH_SIZE,
        current_length=EXAMPLE_CURRENT_LENGTH,
        past_length=EXAMPLE_PAST_LENGTH,
        vocab_size=config.vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_size=head_size,
        seed=SEED,
    )

    example_input_ids = torch.from_numpy(example_input_ids_np)

    example_flat_past = tuple(torch.from_numpy(value) for value in example_flat_past_np)

    # =========================================================================
    # 3. Dynamic dimensions
    # =========================================================================

    batch_dimension = torch.export.Dim(
        "batch_size",
        min=1,
    )

    current_dimension = torch.export.Dim(
        "current_sequence_length",
        min=1,
    )

    past_dimension = torch.export.Dim(
        "past_sequence_length",
        min=1,
    )

    # forward() has two arguments:
    #
    #   input_ids
    #   flat_past
    #
    # flat_past is itself a tuple.
    dynamic_shapes = (
        {
            0: batch_dimension,
            1: current_dimension,
        },
        tuple(
            {
                0: batch_dimension,
                2: past_dimension,
            }
            for _ in example_flat_past
        ),
    )

    # =========================================================================
    # 4. ONNX I/O names
    # =========================================================================

    input_names = [
        "input_ids",
    ]

    output_names = [
        "logits",
    ]

    for layer_index in range(num_layers):
        input_names.extend(
            (
                f"past_key_{layer_index}",
                f"past_value_{layer_index}",
            )
        )

        output_names.extend(
            (
                f"present_key_{layer_index}",
                f"present_value_{layer_index}",
            )
        )

    # =========================================================================
    # 5. Export RAW
    # =========================================================================

    print()
    print("[2] Exporting RAW ONNX")

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            args=(
                example_input_ids,
                example_flat_past,
            ),
            f=RAW_PATH,
            input_names=input_names,
            output_names=output_names,
            opset_version=OPSET,
            dynamo=True,
            optimize=True,
            dynamic_shapes=dynamic_shapes,
            external_data=False,
        )

    raw_model = onnx.load(RAW_PATH)

    onnx.checker.check_model(raw_model)

    # Standard ONNX inference, NOT ORT symbolic inference.
    raw_model = onnx.shape_inference.infer_shapes(raw_model)

    print_statistics(
        "RAW",
        raw_model,
    )

    if count_attention(raw_model) != num_layers:
        raise RuntimeError(
            "Expected "
            f"{num_layers} standard Attention nodes, "
            f"found {count_attention(raw_model)}."
        )

    # =========================================================================
    # 6. Rewrite Attention + cache
    # =========================================================================

    print()
    print("[3] Rewriting Attention + KV cache")

    attention_rules = []

    # Depending on exporter/version, torch.cat(dim=-2)
    # can appear as either -2 or normalized 2.
    for cache_axis in (
        -2,
        2,
    ):
        attention_rules.append(
            make_gpt2_cache_attention_rule(
                num_heads=num_heads,
                cache_axis=cache_axis,
                masked=True,
            )
        )

        attention_rules.append(
            make_gpt2_cache_attention_rule(
                num_heads=num_heads,
                cache_axis=cache_axis,
                masked=False,
            )
        )

    attention_model = onnxscript.rewriter.rewrite(
        raw_model,
        pattern_rewrite_rules=attention_rules,
    )

    print_statistics(
        "AFTER ATTENTION REWRITE",
        attention_model,
    )

    rewritten_attentions = count_3d_cache_attention(attention_model)

    if rewritten_attentions != num_layers:
        raise RuntimeError(
            "Attention rewrite did not match every block: "
            f"expected {num_layers}, "
            f"found {rewritten_attentions}."
        )

    # Save intermediate result because it is useful
    # when inspecting the model with Netron.
    onnx.save(
        attention_model,
        ATTENTION_PATH,
    )

    # =========================================================================
    # 7. Re-run shape inference
    # =========================================================================

    attention_model = onnx.shape_inference.infer_shapes(attention_model)

    # =========================================================================
    # 8. Validate Gemm
    # =========================================================================

    print()
    print("[4] Validating GPT-2 Gemm")

    validate_gpt2_gemms(attention_model)

    gemm_before = sum(node.op_type == "Gemm" for node in attention_model.graph.node)

    # =========================================================================
    # 9. GPT-2 Conv1D canonicalization
    # =========================================================================

    print()
    print("[5] Rewriting Reshape -> Gemm -> Reshape as MatMul -> Add")

    conv1d_rule = make_gpt2_conv1d_rule()

    final_model = onnxscript.rewriter.rewrite(
        attention_model,
        pattern_rewrite_rules=[
            conv1d_rule,
        ],
    )

    gemm_after = sum(node.op_type == "Gemm" for node in final_model.graph.node)

    print(f"Matched GPT-2 Conv1D-like Gemm: {gemm_before - gemm_after}")

    print_statistics(
        "AFTER GPT-2 CONV1D REWRITE",
        final_model,
    )

    # =========================================================================
    # 10. GPT-2 NewGELU canonicalization
    # =========================================================================

    print()
    print("[6] Rewriting GPT-2 NewGELU as standard ONNX Gelu")

    tanh_before = sum(node.op_type == "Tanh" for node in final_model.graph.node)

    gelu_before = sum(
        node.op_type == "Gelu" and node.domain in {"", "ai.onnx"}
        for node in final_model.graph.node
    )

    gelu_rule = make_gpt2_gelu_new_rule()

    gelu_rules = pattern.RewriteRuleSet(
        [
            gelu_rule,
        ],
        commute=True,
    )

    tanh_before = sum(node.op_type == "Tanh" for node in final_model.graph.node)

    gelu_before = sum(node.op_type == "Gelu" for node in final_model.graph.node)

    final_model = onnxscript.rewriter.rewrite(
        final_model,
        pattern_rewrite_rules=gelu_rules,
    )

    tanh_after = sum(node.op_type == "Tanh" for node in final_model.graph.node)

    gelu_after = sum(node.op_type == "Gelu" for node in final_model.graph.node)

    print(f"Tanh: {tanh_before} -> {tanh_after}")

    print(f"Gelu: {gelu_before} -> {gelu_after}")

    final_model = onnxscript.rewriter.rewrite(
        final_model,
        pattern_rewrite_rules=[
            gelu_rule,
        ],
    )

    tanh_after = sum(node.op_type == "Tanh" for node in final_model.graph.node)

    gelu_after = sum(
        node.op_type == "Gelu" and node.domain in {"", "ai.onnx"}
        for node in final_model.graph.node
    )

    print(f"Tanh before : {tanh_before}")

    print(f"Tanh after  : {tanh_after}")

    print(f"Gelu before : {gelu_before}")

    print(f"Gelu after  : {gelu_after}")

    print(f"Fused GELU  : {gelu_after - gelu_before}")

    # =========================================================================
    # 10. Cleanup
    # =========================================================================

    print()
    print("[6] ONNX Script cleanup")

    final_model = onnxscript_optimizer.optimize(
        final_model,
        num_iterations=2,
        onnx_shape_inference=True,
        # ai.onnx::Attention is an ONNX Function.
        # We explicitly want to preserve it.
        inline=False,
    )

    onnx.checker.check_model(final_model)

    print_statistics(
        "FINAL",
        final_model,
    )

    if count_3d_cache_attention(final_model) != num_layers:
        raise RuntimeError("Attention nodes changed unexpectedly during optimization.")

    # =========================================================================
    # 11. Save final
    # =========================================================================

    print()
    print("[7] Saving FINAL")

    onnx.save(
        final_model,
        FINAL_PATH,
    )

    # Test exactly the serialized artifact.
    final_disk = onnx.load(FINAL_PATH)

    onnx.checker.check_model(final_disk)

    # =========================================================================
    # 12. Ensure public interface is unchanged
    # =========================================================================

    raw_disk = onnx.load(RAW_PATH)

    raw_inputs = [value.name for value in raw_disk.graph.input]

    final_inputs = [value.name for value in final_disk.graph.input]

    raw_outputs = [value.name for value in raw_disk.graph.output]

    final_outputs = [value.name for value in final_disk.graph.output]

    if raw_inputs != final_inputs:
        raise RuntimeError("The rewrite changed model inputs.")

    if raw_outputs != final_outputs:
        raise RuntimeError("The rewrite changed model outputs.")

    # =========================================================================
    # 13. Graph comparison
    # =========================================================================

    raw_nodes = len(raw_disk.graph.node)

    attention_nodes = len(onnx.load(ATTENTION_PATH).graph.node)

    final_nodes = len(final_disk.graph.node)

    print()
    print("=" * 100)
    print("GRAPH COMPARISON")
    print("=" * 100)

    print(f"RAW                  : {raw_nodes}")

    print(f"After Attention      : {attention_nodes}")

    print(f"FINAL                : {final_nodes}")

    print(f"Total removed        : {raw_nodes - final_nodes}")

    print(
        f"Total reduction      : {100.0 * (raw_nodes - final_nodes) / raw_nodes:.2f}%"
    )

    # =========================================================================
    # 14. Numerical verification
    # =========================================================================

    print()
    print("[8] Numerical verification")

    verify_models(
        raw_path=RAW_PATH,
        final_path=FINAL_PATH,
        config=config,
    )

    # =========================================================================
    # 15. Cache -> cache execution
    # =========================================================================

    verify_cache_round_trip(
        raw_path=RAW_PATH,
        final_path=FINAL_PATH,
        config=config,
    )

    # =========================================================================
    # Done
    # =========================================================================

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(f"RAW       : {RAW_PATH}")

    print(f"ATTENTION : {ATTENTION_PATH}")

    print(f"FINAL     : {FINAL_PATH}")


if __name__ == "__main__":
    main()
