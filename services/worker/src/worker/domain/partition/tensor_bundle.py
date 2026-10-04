import numpy as np
from pydantic import BaseModel, ConfigDict

from shared.model.keys import TensorKey


class Tensor(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    value: np.ndarray


class TensorBundle(BaseModel):
    model_config = ConfigDict(frozen=True)
    bundle: dict[TensorKey, Tensor]

    def get_tensor_by_name(self, name: str) -> Tensor | None:
        return self.bundle.get(name, None)

    def filter(self, names: list[str]) -> TensorBundle:
        return TensorBundle(
            bundle={
                name: tensor for name, tensor in self.bundle.items() if name in names
            }
        )

    def merge(self, other: TensorBundle) -> TensorBundle:
        return TensorBundle(
            bundle={
                **self.bundle,
                **other.bundle,
            }
        )

    def get_tensor_names(self) -> list[str]:
        return list(self.bundle.keys())
