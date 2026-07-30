from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, model_serializer, model_validator


class Tensor(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    tensor: np.ndarray
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype

    @model_serializer(mode="plain")
    def serialize(self) -> dict[str, Any]:
        array = np.ascontiguousarray(self.tensor)

        return {
            "tensor": array.tobytes(),
            "name": self.name,
            "shape": array.shape,
            # Unlike dtype.name, dtype.str preserves byte order.
            "dtype": array.dtype.str,
        }

    @model_validator(mode="before")
    @classmethod
    def deserialize(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        raw_tensor = value.get("tensor")
        if not isinstance(raw_tensor, (bytes, bytearray, memoryview)):
            return value

        shape = tuple(value["shape"])
        dtype = np.dtype(value["dtype"])

        tensor = np.frombuffer(raw_tensor, dtype=dtype).reshape(shape).copy()

        return {
            **value,
            "tensor": tensor,
            "shape": shape,
            "dtype": dtype,
        }


class TensorBundle(BaseModel):
    model_config = ConfigDict(frozen=True)
    bundle: dict[str, Tensor]
