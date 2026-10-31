from dataclasses import dataclass

import numpy as np

from front_end.domain.tensor_map import TensorMap
from shared.model.model_variant import ModelVariantId

type ShapeDimension = int | str | None


@dataclass(frozen=True, slots=True)
class InputSpec:
    name: str
    shape: tuple[ShapeDimension, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class ModelSpecs:
    model_variant_id: ModelVariantId
    inputs: tuple[InputSpec, ...]

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.inputs]

        if len(names) != len(set(names)):
            raise ValueError("Duplicate input names")

        for spec in self.inputs:
            np.dtype(spec.dtype)

            for dim in spec.shape:
                if isinstance(dim, int) and dim < 0:
                    raise ValueError(f"Invalid dimension {dim} for input '{spec.name}'")
                if isinstance(dim, str) and not dim:
                    raise ValueError("Empty symbolic dimension name")

    def validate_tensor_map(self, tensor_map: TensorMap) -> None:
        expected_names = {spec.name for spec in self.inputs}
        actual_names = set(tensor_map)

        if expected_names != actual_names:
            raise ValueError(
                f"Invalid model inputs: "
                f"missing={expected_names - actual_names}, "
                f"unexpected={actual_names - expected_names}"
            )

        symbolic_dimensions: dict[str, int] = {}

        for spec in self.inputs:
            tensor = tensor_map[spec.name]

            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Input '{spec.name}' must be a NumPy array")

            if tensor.dtype != np.dtype(spec.dtype):
                raise ValueError(
                    f"Invalid dtype for '{spec.name}': "
                    f"expected {spec.dtype}, got {tensor.dtype}"
                )

            if tensor.ndim != len(spec.shape):
                raise ValueError(
                    f"Invalid rank for '{spec.name}': "
                    f"expected {len(spec.shape)}, got {tensor.ndim}"
                )

            for axis, (expected, actual) in enumerate(
                zip(spec.shape, tensor.shape, strict=True)
            ):
                if expected is None:
                    continue

                if isinstance(expected, int):
                    if actual != expected:
                        raise ValueError(
                            f"Invalid dimension for '{spec.name}' "
                            f"at axis {axis}: "
                            f"expected {expected}, got {actual}"
                        )

                elif isinstance(expected, str):
                    if expected in symbolic_dimensions:
                        if symbolic_dimensions[expected] != actual:
                            raise ValueError(
                                f"Inconsistent symbolic dimension "
                                f"'{expected}' for '{spec.name}': "
                                f"expected {symbolic_dimensions[expected]}, "
                                f"got {actual}"
                            )
                    else:
                        symbolic_dimensions[expected] = actual
