from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, model_serializer, model_validator


class Tensor(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    value: np.ndarray
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype

    def get_value(self) -> np.ndarray:
        return self.value

    @model_serializer(mode="plain")
    def serialize(self) -> dict[str, Any]:
        array = np.ascontiguousarray(self.value)

        return {
            "value": array.tobytes(),
            "name": self.name,
            "shape": array.shape,
            # Unlike dtype.name, dtype.str preserves byte order.
            "dtype": array.dtype.str,
        }

    @model_validator(mode="before")
    @classmethod
    def deserialize(cls, des_whole: Any) -> Any:
        if not isinstance(des_whole, dict):
            return des_whole

        raw_value = des_whole.get("value")
        if not isinstance(raw_value, (bytes, bytearray, memoryview)):
            return des_whole

        shape = tuple(des_whole["shape"])
        dtype = np.dtype(des_whole["dtype"])

        value = np.frombuffer(raw_value, dtype=dtype).reshape(shape).copy()

        return {
            **des_whole,
            "value": value,
            "shape": shape,
            "dtype": dtype,
        }


class TensorBundle(BaseModel):
    model_config = ConfigDict(frozen=True)
    bundle: dict[str, Tensor]

    def get_tensor_by_name(self, name: str) -> Tensor | None:
        return self.bundle.get(name, None)
