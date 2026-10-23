from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class Tensor:
    """Tensor value with an encapsulated storage representation."""

    _value: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self._value, np.ndarray):
            raise TypeError("Tensor value must be a NumPy array")
        owned = np.array(self._value, copy=True)
        owned.flags.writeable = False
        object.__setattr__(
            self,
            "_value",
            owned,
        )

    @classmethod
    def from_numpy(cls, value: np.ndarray) -> Self:
        return cls(_value=value)

    def as_numpy(self) -> np.ndarray:
        view = self._value.view()
        view.flags.writeable = False
        return view
