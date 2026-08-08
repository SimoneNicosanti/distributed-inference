from pathlib import Path

import onnx
import onnxscript
from onnxscript import ir
from onnxscript.rewriter import pattern

RAW_PATH = Path("vit_raw.onnx")
REWRITTEN_PATH = Path("vit_attention_3d.onnx")

NUM_HEADS = 12


def make_attention_4d_to_3d_rule(
    num_heads: int,
) -> pattern.RewriteRule:
    # -------------------------------------------------------------------------
    # Target:
    #
    # q3 -> Reshape -> Transpose --\
    # k3 -> Reshape -> Transpose ---- Attention -> Transpose -> Reshape
    # v3 -> Reshape -> Transpose --/
    #
    # Reshape converts:
    #
    #   [B, S, H*D]
    #
    # into:
    #
    #   [B, S, H, D]
    #
    # and Transpose converts that into:
    #
    #   [B, H, S, D]
    # -------------------------------------------------------------------------

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
        )

        attention_transposed = op.Transpose(
            attention,
            perm=[0, 2, 1, 3],
        )

        return op.Reshape(
            attention_transposed,
            output_shape,
        )

    # -------------------------------------------------------------------------
    # Replacement:
    #
    # q3 --\
    # k3 --- Attention3D
    # v3 --/
    #
    # The standard ONNX Attention performs the head split/merge internally.
    # -------------------------------------------------------------------------

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
            _domain="",
        )

    # -------------------------------------------------------------------------
    # Safety check.
    #
    # For this first version we only accept the transformation when the
    # pre-attention tensors are known to be rank 3.
    # -------------------------------------------------------------------------

    def match_condition(
        context,
        q: ir.Value,
        k: ir.Value,
        v: ir.Value,
        **_,
    ) -> bool:
        del context

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


model = onnx.load(RAW_PATH)

rule = make_attention_4d_to_3d_rule(NUM_HEADS)

model = onnxscript.rewriter.rewrite(
    model,
    pattern_rewrite_rules=[
        rule,
    ],
)

# model = onnxscript_optimizer.optimize(
#     model,
#     inline=False,
#     onnx_shape_inference=True,
# )

onnx.checker.check_model(model)

onnx.save(
    model,
    REWRITTEN_PATH,
)


import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(str(REWRITTEN_PATH))

pixel_values = ort.OrtValue.ortvalue_from_numpy(
    np.random.randn(
        1,
        3,
        224,
        224,
    ).astype(np.float32)
)

outputs = session.run(
    None,
    {
        "pixel_values": pixel_values,
    },
)

print(outputs[0].shape)
