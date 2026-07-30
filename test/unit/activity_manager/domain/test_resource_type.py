import pytest
from pydantic import ValidationError

from distributed_inference.activity_manager.domain.resource_type import (
    ResourceRequirement,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("quantity", "exclusive"),
    [
        (1.0, True),
        (0.0, False),
        (-1.0, False),
        (float("nan"), False),
        (float("inf"), False),
        (float("-inf"), False),
    ],
)
def test_resource_requirement_rejects_inconsistent_quantity(
    quantity: float,
    exclusive: bool,
) -> None:
    with pytest.raises(ValidationError):
        ResourceRequirement(quantity=quantity, exclusive=exclusive)


@pytest.mark.unit
def test_resource_requirement_accepts_shared_and_exclusive_forms() -> None:
    shared = ResourceRequirement(quantity=2.5, exclusive=False)
    exclusive = ResourceRequirement(quantity=0, exclusive=True)

    assert shared.quantity == 2.5
    assert not shared.exclusive
    assert exclusive.quantity == 0
    assert exclusive.exclusive
