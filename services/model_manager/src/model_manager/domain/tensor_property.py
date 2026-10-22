from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorProperty:
    data_type: str
    data_type_size_in_bytes: int
    shape_expression: tuple[int | str, ...]

    def __post_init__(self) -> None:
        if self.data_type_size_in_bytes <= 0:
            raise ValueError("Data type size must be positive")
