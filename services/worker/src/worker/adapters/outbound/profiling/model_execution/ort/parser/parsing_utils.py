import json
import math
from collections import defaultdict
from pathlib import Path
from typing import cast

import numpy as np

from shared.model.keys import LayerKey
from worker.domain.profiling.model_execution.model_execution_profile import (
    LayerExecutionProfile,
)

NODE_EVENT_CATEGORY = "Node"
NODE_EVENT_SUFFIX = "_kernel_time"
MEMORY_BYTES_IN_USE_ARGUMENT = "mem_bytes_in_use"
PROFILE_DURATION_MICROSECONDS_PER_SECOND = 1_000_000


def read_profile_events(profile_path: Path) -> list[dict[str, object]]:
    parsed_profile = cast(object, json.loads(profile_path.read_text()))
    if not isinstance(parsed_profile, list):
        raise ValueError("ONNX Runtime profile must contain a JSON array")
    if not all(isinstance(event, dict) for event in parsed_profile):
        raise ValueError("ONNX Runtime profile contains a non-object event")

    return cast(list[dict[str, object]], parsed_profile)


def build_layer_profiles(
    durations_by_layer: dict[LayerKey, list[float | None]],
    profile_events: list[dict[str, object]],
    providers: set[str],
    *,
    warmup_iterations: int,
    profile_iterations: int,
) -> dict[LayerKey, LayerExecutionProfile]:
    expected_samples = warmup_iterations + profile_iterations
    for layer_key, durations in durations_by_layer.items():
        if len(durations) != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} profiling samples for "
                f"layer {layer_key!r}, got {len(durations)}"
            )

    # Keep invocation indices aligned across every layer before selecting samples.
    valid_run_indices = {
        index
        for index in range(warmup_iterations, expected_samples)
        if durations_by_layer
        and all(
            durations[index] is not None for durations in durations_by_layer.values()
        )
    }
    if not valid_run_indices:
        raise ValueError("No valid profiling runs collected")

    memory_bytes_by_layer = parse_memory_bytes_in_use_by_layer(
        profile_events, providers, valid_run_indices=valid_run_indices
    )
    layer_profiles: dict[LayerKey, LayerExecutionProfile] = {}
    for layer_key, durations in durations_by_layer.items():
        memory_samples = memory_bytes_by_layer.get(layer_key, [])
        if len(memory_samples) != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} memory samples for "
                f"layer {layer_key!r}, got {len(memory_samples)}"
            )

        measured_durations = [
            duration
            for index, duration in enumerate(durations)
            if index in valid_run_indices and duration is not None
        ]
        measured_memory_samples = [
            memory_bytes
            for index, memory_bytes in enumerate(memory_samples)
            if index in valid_run_indices
        ]
        available_memory_samples = [
            memory_bytes
            for memory_bytes in measured_memory_samples
            if memory_bytes is not None
        ]
        if available_memory_samples and len(available_memory_samples) != len(
            measured_memory_samples
        ):
            raise ValueError(f"Inconsistent memory samples for layer {layer_key!r}")

        layer_profiles[layer_key] = LayerExecutionProfile(
            execution_time_s=float(np.mean(measured_durations)),
            memory_bytes=(
                max(available_memory_samples) if available_memory_samples else None
            ),
        )

    return layer_profiles


def parse_memory_bytes_in_use_by_layer(
    profile_events: list[dict[str, object]],
    providers: set[str],
    *,
    valid_run_indices: set[int],
) -> dict[LayerKey, list[int | None]]:
    memory_bytes_by_layer: dict[LayerKey, list[int | None]] = defaultdict(list)

    for event in profile_events:
        event_arguments = event.get("args")
        if not isinstance(event_arguments, dict):
            continue

        provider = event_arguments.get("provider")
        if not isinstance(provider, str) or provider not in providers:
            continue

        layer_key = node_layer_key_for_provider(event, provider)
        if layer_key is not None:
            samples = memory_bytes_by_layer[layer_key]
            samples.append(
                event_memory_bytes_in_use(event)
                if len(samples) in valid_run_indices
                else None
            )

    return dict(memory_bytes_by_layer)


def event_memory_bytes_in_use(event: dict[str, object]) -> int | None:
    event_arguments = cast(dict[object, object], event["args"])
    raw_memory_bytes = event_arguments.get(MEMORY_BYTES_IN_USE_ARGUMENT)
    if raw_memory_bytes is None:
        return None

    if isinstance(raw_memory_bytes, bool) or not isinstance(
        raw_memory_bytes, (int, str)
    ):
        raise ValueError("ONNX Runtime profile has invalid memory bytes in use")

    try:
        memory_bytes = int(raw_memory_bytes)
    except ValueError as error:
        raise ValueError(
            "ONNX Runtime profile has invalid memory bytes in use"
        ) from error

    if memory_bytes < 0:
        raise ValueError("ONNX Runtime profile has negative memory bytes in use")

    return memory_bytes


def node_layer_key_for_provider(
    event: dict[str, object], provider: str
) -> LayerKey | None:
    event_name = event.get("name")
    event_arguments = event.get("args")
    if (
        event.get("cat") != NODE_EVENT_CATEGORY
        or not isinstance(event_name, str)
        or not event_name.endswith(NODE_EVENT_SUFFIX)
        or not isinstance(event_arguments, dict)
        or event_arguments.get("provider") != provider
    ):
        return None

    return event_name.removesuffix(NODE_EVENT_SUFFIX)


def event_duration_seconds(layer: str, event: dict[str, object]) -> float | None:
    """Return seconds, or None when a negative duration invalidates the run."""
    duration = event.get("dur")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
    ):
        raise ValueError(
            f"ONNX Runtime profile event for layer {layer} has an invalid duration"
        )
    if duration < 0:
        return None

    return float(duration) / PROFILE_DURATION_MICROSECONDS_PER_SECOND
