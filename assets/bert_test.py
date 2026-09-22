import torch
import torch.nn as nn
from onnxscript.onnx_opset import opset23 as op
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "bert-base-uncased"


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
    scale: float | None = None,
    enable_gqa: bool = False,
):
    # In model.eval() BERT usa dropout=0 nell'attention.
    if dropout_p != 0.0:
        raise ValueError("Only inference with dropout_p=0 is supported")

    if enable_gqa:
        raise ValueError("BERT uses regular MHA, not GQA")

    kwargs = {
        "is_causal": int(is_causal),
    }

    if scale is not None:
        kwargs["scale"] = scale

    if attn_mask is None:
        outputs = op.Attention(
            query,
            key,
            value,
            **kwargs,
        )
    else:
        outputs = op.Attention(
            query,
            key,
            value,
            attn_mask,
            **kwargs,
        )

    # Attention has optional present-K/V and QK outputs.
    # SDPA wants only Y.
    return outputs[0]


# ---------------------------------------------------------
# BERT
# ---------------------------------------------------------

bert = AutoModel.from_pretrained(
    MODEL_ID,
    attn_implementation="sdpa",
)

bert.eval()


class BertWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state


model = BertWrapper(bert).eval()


# ---------------------------------------------------------
# Example input
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

inputs = tokenizer(
    "The quick brown fox jumps over the lazy dog.",
    return_tensors="pt",
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
token_type_ids = inputs.get(
    "token_type_ids",
    torch.zeros_like(input_ids),
)


# ---------------------------------------------------------
# Export
# ---------------------------------------------------------

batch = torch.export.Dim("batch", min=1, max=32)
seq = torch.export.Dim("sequence_length", min=1, max=512)

dynamic_shapes = {
    "input_ids": {
        0: batch,
        1: seq,
    },
    "attention_mask": {
        0: batch,
        1: seq,
    },
    "token_type_ids": {
        0: batch,
        1: seq,
    },
}

onnx_program = torch.onnx.export(
    model,
    (
        input_ids,
        attention_mask,
        token_type_ids,
    ),
    opset_version=23,
    dynamo=True,
    input_names=[
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ],
    output_names=["last_hidden_state"],
    dynamic_shapes=dynamic_shapes,
    custom_translation_table={
        torch.ops.aten.scaled_dot_product_attention.default: sdpa_to_onnx_attention,
    },
    optimize=True,
)

onnx_program.save("bert_base_dynamic.onnx")
