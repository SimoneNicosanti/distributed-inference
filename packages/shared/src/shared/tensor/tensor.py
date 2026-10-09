import numpy as np
from pydantic import BaseModel, ConfigDict


class Tensor(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
    )

    value: np.ndarray
