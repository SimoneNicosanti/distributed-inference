from collections.abc import Mapping

import numpy as np
from pydantic import BaseModel, ConfigDict

from integration.dto.tensor.tensor_dto import TensorDto


class TensorBundleDto(BaseModel):
    model_config = ConfigDict(frozen=True)

    bundle: dict[str, TensorDto]

    @classmethod
    def from_ndarrays(
        cls,
        bundle: Mapping[str, np.ndarray],
    ) -> TensorBundleDto:
        return cls(
            bundle={
                name: TensorDto.from_ndarray(value) for name, value in bundle.items()
            }
        )

    def to_ndarrays(self) -> dict[str, np.ndarray]:
        return {
            name: tensor_dto.to_ndarray()
            for name, tensor_dto in self.bundle.items()
        }
