import torch
import torch.nn as nn
import onnx

from transformers import ViTForImageClassification
from onnxscript.onnx_opset import opset23 as op


# ------------------------------------------------------------
# aten.scaled_dot_product_attention -> ai.onnx::Attention
# ------------------------------------------------------------


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
    # Per inference ViT questo deve essere 0.
    if dropout_p != 0.0:
        raise ValueError("This lowering only supports dropout_p=0")

    # ViT usa normale MHA, non GQA.
    if enable_gqa:
        raise ValueError("GQA not expected for ViT")

    kwargs = {
        "is_causal": int(is_causal),
    }

    if scale is not None:
        kwargs["scale"] = scale

    if attn_mask is None:
        y, _, _, _ = op.Attention(
            query,
            key,
            value,
            **kwargs,
        )
    else:
        y, _, _, _ = op.Attention(
            query,
            key,
            value,
            attn_mask,
            **kwargs,
        )

    return y


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    attn_implementation="sdpa",
)

model.eval()


# Evitiamo output HuggingFace complessi.
class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


model = Wrapper(model)

x = torch.randn(1, 3, 224, 224)


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------

onnx_program = torch.onnx.export(
    model,
    (x,),
    opset_version=23,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamo=True,
    custom_translation_table={
        torch.ops.aten.scaled_dot_product_attention.default: sdpa_to_onnx_attention,
    },
    # Per il primo test eviterei altre trasformazioni.
    optimize=True,
)

onnx_program.save("vit_attention.onnx")
