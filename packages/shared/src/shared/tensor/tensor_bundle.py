from pydantic import BaseModel, ConfigDict

from shared.model.keys import TensorKey
from shared.tensor.tensor import Tensor


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
