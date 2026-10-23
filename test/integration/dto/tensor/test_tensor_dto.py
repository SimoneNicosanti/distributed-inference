import numpy as np

from integration.dto.tensor.tensor_bundle_dto import TensorBundleDto
from integration.dto.tensor.tensor_dto import TensorDto


def test_tensor_dto_round_trips_ndarray() -> None:
    value = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)

    restored = TensorDto.from_ndarray(value).to_ndarray()

    np.testing.assert_array_equal(restored, value)
    assert restored.dtype == value.dtype


def test_tensor_bundle_dto_round_trips_ndarrays() -> None:
    bundle = {
        "input_ids": np.array([[1, 2]], dtype=np.int64),
        "attention_mask": np.array([[True, False]], dtype=np.bool_),
    }

    restored = TensorBundleDto.from_ndarrays(bundle).to_ndarrays()

    assert restored.keys() == bundle.keys()
    for name, value in bundle.items():
        np.testing.assert_array_equal(restored[name], value)
        assert restored[name].dtype == value.dtype
