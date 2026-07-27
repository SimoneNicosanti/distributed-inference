from pydantic import BaseModel, ConfigDict


## This is the input for the local sub-model inference
class InferenceInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    pass


## This is the output of the local sub-model inference
class InferenceOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    pass
