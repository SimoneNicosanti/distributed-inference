BYTES_PER_MEGABYTE = 1_000_000.0


def bytes_to_megabytes(value: int | float) -> float:
    return float(value) / BYTES_PER_MEGABYTE


def bytes_per_second_to_megabytes_per_second(value: int | float) -> float:
    return float(value) / BYTES_PER_MEGABYTE
