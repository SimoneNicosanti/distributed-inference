import numpy as np

from worker.domain.tensor.tensor import Tensor
from worker.domain.tensor.tensor_bundle import TensorBundle


def test_tensor_exposes_numpy_representation_through_explicit_adapter() -> None:
    value = np.array([1, 2, 3], dtype=np.float32)

    tensor = Tensor.from_numpy(value)

    np.testing.assert_array_equal(tensor.as_numpy(), value)


def test_tensor_bundle_filters_and_merges_tensors() -> None:
    first = Tensor.from_numpy(np.array([1]))
    replacement = Tensor.from_numpy(np.array([2]))
    second = Tensor.from_numpy(np.array([3]))
    bundle = TensorBundle(bundle={"first": first})

    filtered = bundle.filter(["first"])
    merged = bundle.merge(
        TensorBundle(bundle={"first": replacement, "second": second})
    )

    assert filtered.bundle == {"first": first}
    assert merged.bundle == {"first": replacement, "second": second}
    assert merged.get_tensor_by_name("second") is second
    assert merged.get_tensor_names() == {"first", "second"}
