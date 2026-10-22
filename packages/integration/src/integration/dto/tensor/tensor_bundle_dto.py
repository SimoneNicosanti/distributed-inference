from pydantic import BaseModel, ConfigDict

from integration.dto.tensor.tensor_dto import TensorDto
from shared.tensor.tensor_bundle import TensorBundle


class TensorBundleDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    bundle: dict[str, TensorDto]

    @classmethod
    def from_tensor_bundle(cls, bundle: TensorBundle) -> TensorBundleDto:
        return cls(
            bundle={
                name: TensorDto.from_tensor(tensor)
                for name, tensor in bundle.bundle.items()
            }
        )

    def to_tensor_bundle(self) -> TensorBundle:
        return TensorBundle(
            bundle={
                name: tensor_dto.to_tensor() for name, tensor_dto in self.bundle.items()
            }
        )
