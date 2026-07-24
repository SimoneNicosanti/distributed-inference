from pathlib import Path

from pydantic import BaseModel, ConfigDict

from distributed_inference.domain.model_graph_info import ModelGraph


class ModelAssetInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    sha256: str
    size_bytes: int
    opset: int


class AnalyzedModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact: ModelAssetInfo
    model_graph: ModelGraph
