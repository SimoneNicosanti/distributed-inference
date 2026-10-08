from pydantic import BaseModel, ConfigDict, PositiveInt


class ShapePoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_shape_points: dict[str, tuple[PositiveInt, ...]]
