from pydantic import BaseModel, ConfigDict


class TensorProperty(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_type: str
    data_type_size_in_bytes: int
    shape_expression: tuple[int | str, ...]
