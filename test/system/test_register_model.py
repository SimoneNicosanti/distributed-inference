import os
from uuid import uuid4

import httpx
import pytest


@pytest.mark.integration
def test_register_model() -> None:
    base_url = os.getenv("FRONT_END_BASE_URL")
    if base_url is None:
        pytest.skip("Set FRONT_END_BASE_URL to run system tests")

    unique_id = uuid4()
    request = {
        "model_dto": {
            "model_id": {
                "model_name": f"system-test-{unique_id.hex}",
            },
            "model_task": "classification",
            "architecture_info": {"kind": "cnn"},
        }
    }

    response = httpx.post(
        f"{base_url.rstrip('/')}/front-end/models",
        json=request,
        timeout=10.0,
    )

    assert response.status_code == 200, response.text
