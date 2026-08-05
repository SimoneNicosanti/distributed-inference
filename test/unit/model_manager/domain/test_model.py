import pytest
from pydantic import ValidationError

from distributed_inference.model_manager.domain.model import (
    ModelId,
    ModelTask,
    ModelType,
)
from test.support.model_manager.model_domain_test_utils import (
    build_model,
    build_model_id,
    build_model_info,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_name", "message"),
    [
        ("owner/resnet50", "cannot contain '/'"),
        ("owner\\resnet50", "cannot contain '\\\\'"),
        ("../resnet50", "cannot contain"),
    ],
)
def test_model_id_rejects_path_traversal_in_the_model_name(
    model_name: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ModelId(owner_id=build_model_id().owner_id, model_name=model_name)


@pytest.mark.unit
def test_model_exposes_the_identity_of_its_model_id() -> None:
    model_id = build_model_id(model_name="resnet50")
    model = build_model(
        model_id=model_id,
        model_info=build_model_info(
            model_task=ModelTask.DETECTION,
            model_type=ModelType.VIT,
        ),
    )

    assert model.model_name == "resnet50"
    assert model.owner_id == model_id.owner_id
    assert model.model_info.model_task == ModelTask.DETECTION
    assert model.model_info.model_type == ModelType.VIT
