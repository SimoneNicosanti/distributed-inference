"""Independent, addressable streams: adding entities cannot shift old draws."""

import hashlib
import json
import random
from uuid import UUID


def fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def stream(seed: int, namespace: str, *keys: str) -> random.Random:
    return random.Random(int(fingerprint([seed, namespace, *keys]), 16))


def stable_uuid(seed: int, namespace: str, key: str) -> UUID:
    return UUID(fingerprint([seed, namespace, key])[:32], version=5)
