from typing import Any, List

import numpy as np
import onnx_tool
from onnx_tool.node import DIV_MACS, EXP_MACS, Node
from onnx_tool.tensor import Tensor
from onnx_tool.utils import NODE_REGISTRY


@NODE_REGISTRY.register()  # type: ignore
class DynamicQuantizeLinearNode(onnx_tool.Node):
    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]) -> None:
        input_shape = intensors[0].get_shape()

        # y: tensore quantizzato, stessa shape dell'input
        outtensors[0].update_shape(input_shape)
        outtensors[0].update_dtype(np.uint8)

        # y_scale: scalare float32
        outtensors[1].update_shape([])
        outtensors[1].update_dtype(np.float32)

        # y_zero_point: scalare uint8
        outtensors[2].update_shape([])
        outtensors[2].update_dtype(np.uint8)


@NODE_REGISTRY.register()  # type: ignore
class MultiHeadAttentionNode(Node):
    """Profiler per com.microsoft::MultiHeadAttention senza KV-cache."""

    def __init__(self, node_proto: Any) -> None:
        super().__init__(node_proto)

        if node_proto.domain != "com.microsoft":
            raise ValueError(
                f"MultiHeadAttention domain non supportato: {node_proto.domain!r}"
            )

        if not hasattr(self, "num_heads"):
            raise ValueError(f"num_heads mancante nel nodo {self.name}")

    def shape_infer(
        self,
        intensors: list[Tensor],
        outtensors: list[Tensor],
    ) -> None:
        query_shape = list(intensors[0].get_shape())

        if len(query_shape) != 3:
            raise NotImplementedError(
                f"Supportato solo query [B,S,D], ricevuto "
                f"{query_shape} nel nodo {self.name}"
            )

        batch_size, query_length, query_hidden_size = query_shape

        # Nel caso ordinario:
        # query: [B, Sq, Dq]
        # key:   [B, Sk, Dk]
        # value: [B, Sk, Dv]
        if len(intensors) >= 3:
            value_shape = list(intensors[2].get_shape())
            value_hidden_size = value_shape[-1]
        else:
            value_hidden_size = query_hidden_size

        outtensors[0].update_shape(
            [
                batch_size,
                query_length,
                value_hidden_size,
            ]
        )
        outtensors[0].update_dtype(intensors[0].dtype)

        # Il BERT encoder non dovrebbe produrre KV-cache.
        if len(outtensors) > 1:
            raise NotImplementedError(
                f"Output KV-cache non ancora supportati per {self.name}"
            )

    def _profile_core(
        self,
        intensors: list[Tensor],
        outtensors: list[Tensor],
    ) -> int:
        query_shape = list(intensors[0].get_shape())
        batch_size, query_length, query_hidden_size = query_shape

        if len(intensors) >= 2:
            key_shape = list(intensors[1].get_shape())
            key_length = key_shape[1]
        else:
            key_length = query_length

        if len(intensors) >= 3:
            value_shape = list(intensors[2].get_shape())
            value_hidden_size = value_shape[-1]
        else:
            value_hidden_size = query_hidden_size

        num_heads = 12  # int(self.num_heads)

        # Q @ K^T
        qk_macs = batch_size * query_length * key_length * query_hidden_size

        # Softmax. Coerente con il modello di costo di onnx-tool.
        softmax_macs = (
            batch_size * num_heads * query_length * key_length * (EXP_MACS + DIV_MACS)
        )

        # Attention probabilities @ V
        av_macs = batch_size * query_length * key_length * value_hidden_size

        return qk_macs + softmax_macs + av_macs
