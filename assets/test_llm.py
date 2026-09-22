import torch
import torch.nn as nn
from onnxscript.onnx_opset import opset23 as op
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


# ---------------------------------------------------------
# aten.scaled_dot_product_attention -> ai.onnx::Attention
# ---------------------------------------------------------


def sdpa_to_onnx_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale=None,
    enable_gqa: bool = False,
):
    if dropout_p != 0.0:
        raise ValueError("Only inference with dropout=0 is supported")

    kwargs = {
        "is_causal": int(is_causal),
    }

    if scale is not None:
        kwargs["scale"] = scale

    if attn_mask is None:
        return op.Attention(
            query,
            key,
            value,
            **kwargs,
        )[0]

    return op.Attention(
        query,
        key,
        value,
        attn_mask,
        **kwargs,
    )[0]


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

llm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    attn_implementation="sdpa",
    torch_dtype=torch.float32,
).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]


model = Wrapper(llm).eval()


# ---------------------------------------------------------
# Example input
# ---------------------------------------------------------

inputs = tokenizer(
    "ONNX Runtime is useful for",
    return_tensors="pt",
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]


# ---------------------------------------------------------
# Dynamic dimensions
# ---------------------------------------------------------

batch = torch.export.Dim(
    "batch",
    min=1,
    max=8,
)

seq = torch.export.Dim(
    "sequence_length",
    min=1,
    max=512,
)

dynamic_shapes = {
    "input_ids": {
        0: batch,
        1: seq,
    },
    "attention_mask": {
        0: batch,
        1: seq,
    },
}


# ---------------------------------------------------------
# Export
# ---------------------------------------------------------

onnx_program = torch.onnx.export(
    model,
    (input_ids, attention_mask),
    opset_version=23,
    dynamo=True,
    optimize=True,
    input_names=[
        "input_ids",
        "attention_mask",
    ],
    output_names=[
        "logits",
    ],
    dynamic_shapes=dynamic_shapes,
    custom_translation_table={
        torch.ops.aten.scaled_dot_product_attention.default: sdpa_to_onnx_attention,
    },
)

onnx_program.save("smollm2_attention_dynamic.onnx")


# ---------------------------------------------------------
# ORT optimized model
# ---------------------------------------------------------

import onnxruntime as ort

so = ort.SessionOptions()

so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

so.optimized_model_filepath = "smollm2_attention_dynamic_opt.onnx"

sess = ort.InferenceSession(
    "smollm2_attention_dynamic.onnx",
    sess_options=so,
    providers=["CPUExecutionProvider"],
)


# ---------------------------------------------------------
# Show actual ONNX input shapes
# ---------------------------------------------------------

print("\nONNX inputs:")

for inp in sess.get_inputs():
    print(
        inp.name,
        inp.shape,
        inp.type,
    )


# ---------------------------------------------------------
# Test different shapes
# ---------------------------------------------------------

import numpy as np

vocab_size = llm.config.vocab_size

for batch_size, seq_len in [
    (1, 8),
    (1, 32),
    (2, 16),
    (4, 64),
]:
    test_input_ids = np.random.randint(
        0,
        vocab_size,
        size=(batch_size, seq_len),
        dtype=np.int64,
    )

    test_attention_mask = np.ones(
        (batch_size, seq_len),
        dtype=np.int64,
    )

    outputs = sess.run(
        None,
        {
            "input_ids": test_input_ids,
            "attention_mask": test_attention_mask,
        },
    )

    print(f"B={batch_size}, S={seq_len} -> logits={outputs[0].shape}")
