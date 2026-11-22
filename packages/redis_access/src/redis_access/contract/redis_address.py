from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HashMapRedisAddress:
    hash_map_key: str
    field_key: str


@dataclass(frozen=True, slots=True)
class KeyRedisAddress:
    key: str


type RedisAddress = HashMapRedisAddress | KeyRedisAddress
