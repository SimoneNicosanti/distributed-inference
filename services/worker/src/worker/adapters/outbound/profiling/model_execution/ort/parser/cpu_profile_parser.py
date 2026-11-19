from collections import defaultdict

from shared.model.keys import LayerKey
from worker.adapters.outbound.profiling.model_execution.ort.parser.parsing_utils import (
    build_layer_profiles,
    event_duration_seconds,
    node_layer_key_for_provider,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    Backend,
    BackendExecutionProfile,
)


def parse_cpu_profile(
    profile_events: list[dict[str, object]],
    *,
    warmup_iterations: int,
    profile_iterations: int,
) -> BackendExecutionProfile:
    durations_by_layer = parse_cpu_durations_by_layer(
        profile_events, warmup_iterations=warmup_iterations
    )
    return BackendExecutionProfile(
        backend=Backend.CPU,
        layer_profiles=build_layer_profiles(
            durations_by_layer,
            profile_events,
            {"CPUExecutionProvider"},
            warmup_iterations=warmup_iterations,
            profile_iterations=profile_iterations,
        ),
    )


def parse_cpu_durations_by_layer(
    profile_events: list[dict[str, object]],
    *,
    warmup_iterations: int,
) -> dict[LayerKey, list[float | None]]:
    durations_by_layer: dict[LayerKey, list[float | None]] = defaultdict(list)
    for event in profile_events:
        layer_key = node_layer_key_for_provider(event, "CPUExecutionProvider")
        if layer_key is not None:
            samples = durations_by_layer[layer_key]
            samples.append(
                event_duration_seconds(layer_key, event)
                if len(samples) >= warmup_iterations
                else None
            )

    return dict(durations_by_layer)
