import pytest

from controller.adapters.outbound.optimization.gurobi_units import (
    bytes_per_second_to_megabytes_per_second,
    bytes_to_megabytes,
)


def test_solver_unit_conversions() -> None:
    assert bytes_to_megabytes(3_000_000_000) == pytest.approx(3_000)
    assert bytes_per_second_to_megabytes_per_second(
        3_000_000_000
    ) == pytest.approx(3_000)
