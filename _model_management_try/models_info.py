from pathlib import Path

from distributed_inference.domain.model_graph_info import (
    DynamicShapeType,
    ModelInfo,
    ModelType,
    TaskType,
)

BASE_MODEL_PATH = Path("/workspace/distributed-inference/assets/models")


TEST_MODELS_INFO = {
    # "yolos": ModelInfo(
    #     name="yolos",
    #     accuracy=0.0,
    #     task=TaskType.DETECTION,
    #     type=ModelType.VIT,
    #     dynamic_shapes={"batch_size": DynamicShapeType.BATCH},
    # ),
    "yolo11": ModelInfo(
        name="yolo11x",
        accuracy=0.0,
        task=TaskType.DETECTION,
        type=ModelType.CNN,
        dynamic_shapes={},
    ),
    "vit-base": ModelInfo(
        name="vit_b",
        accuracy=0.0,
        task=TaskType.CLASSIFICATION,
        type=ModelType.VIT,
        dynamic_shapes={"batch_size": DynamicShapeType.BATCH},
        num_heads=12,
        hidden_size=768,
    ),
    "resnet50": ModelInfo(
        name="resnet50",
        accuracy=0.0,
        task=TaskType.CLASSIFICATION,
        type=ModelType.CNN,
        dynamic_shapes={},
    ),
    "bert": ModelInfo(
        name="bert",
        accuracy=0.0,
        task=TaskType.CLASSIFICATION,
        type=ModelType.BERT,
        dynamic_shapes={
            "batch_size": DynamicShapeType.BATCH,
            "sequence_length": DynamicShapeType.SEQUENCE,
        },
        sequence_sizes=[16, 32, 64, 128, 256],
        num_heads=12,
        hidden_size=768,
    ),
}


def get_model_info(model_name: str) -> ModelInfo:
    if model_name.startswith("yolo11"):
        return TEST_MODELS_INFO["yolo11"]
    elif model_name.startswith("vit-base"):
        return TEST_MODELS_INFO["vit-base"]
    elif model_name.startswith("resnet"):
        return TEST_MODELS_INFO["resnet50"]
    elif model_name.startswith("bert"):
        return TEST_MODELS_INFO["bert"]
    else:
        raise ValueError(f"Unknown model name: {model_name}")
