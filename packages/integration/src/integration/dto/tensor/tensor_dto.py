import base64
from io import BytesIO

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from shared.tensor.tensor import Tensor


class TensorDto(BaseModel):
    """Portable NumPy tensor encoded as a base64 ``.npy`` payload."""

    model_config = ConfigDict(frozen=True)

    npy_base64: str = Field(min_length=1)

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> TensorDto:
        buffer = BytesIO()
        np.save(buffer, tensor.value, allow_pickle=False)
        return cls(npy_base64=base64.b64encode(buffer.getvalue()).decode("ascii"))

    def to_tensor(self) -> Tensor:
        encoded_value = self.npy_base64.encode("ascii")
        raw_value = base64.b64decode(encoded_value, validate=True)
        value = np.load(BytesIO(raw_value), allow_pickle=False)

        if not isinstance(value, np.ndarray):
            raise ValueError("Tensor payload does not contain a NumPy array")

        return Tensor(value=value)
