from pydantic import BaseModel, ConfigDict, PositiveFloat


class StreamInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    arrival_rate: PositiveFloat
