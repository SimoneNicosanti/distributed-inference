from collections import defaultdict

from shared.model.keys import LayerKey
from worker.adapters.outbound.profiling.model_execution.ort.parser.parsing_utils import (
    NODE_EVENT_CATEGORY,
    build_layer_profiles,
    event_duration_seconds,
    node_layer_key_for_provider,
)
from worker.domain.profiling.model_execution.model_execution_profile import (
    Backend,
    BackendExecutionProfile,
)

KERNEL_EVENT_CATEGORY = "Kernel"


def parse_gpu_profile(
    profile_events: list[dict[str, object]],
    *,
    warmup_iterations: int,
    profile_iterations: int,
) -> BackendExecutionProfile:
    durations_by_layer = parse_gpu_durations_by_layer(
        profile_events, warmup_iterations=warmup_iterations
    )
    return BackendExecutionProfile(
        backend=Backend.GPU,
        layer_profiles=build_layer_profiles(
            durations_by_layer,
            profile_events,
            {"CUDAExecutionProvider", "CPUExecutionProvider"},
            warmup_iterations=warmup_iterations,
            profile_iterations=profile_iterations,
        ),
    )


def parse_gpu_durations_by_layer(
    profile_events: list[dict[str, object]],
    *,
    warmup_iterations: int,
) -> dict[LayerKey, list[float | None]]:
    durations_by_layer: dict[LayerKey, list[float | None]] = defaultdict(list)
    current_layer_key: LayerKey | None = None
    current_duration: float | None = 0.0
    found_cuda_node = False
    found_kernel_event = False

    def append_current_sample() -> None:
        if current_layer_key is None:
            return
        samples = durations_by_layer[current_layer_key]
        samples.append(current_duration if len(samples) >= warmup_iterations else None)

    for event in profile_events:
        if event.get("cat") == NODE_EVENT_CATEGORY:
            append_current_sample()
            current_layer_key = node_layer_key_for_provider(
                event, "CUDAExecutionProvider"
            )
            current_duration = 0.0
            if current_layer_key is not None:
                found_cuda_node = True
                continue

            cpu_layer_key = node_layer_key_for_provider(event, "CPUExecutionProvider")
            if cpu_layer_key is not None:
                samples = durations_by_layer[cpu_layer_key]
                samples.append(
                    event_duration_seconds(cpu_layer_key, event)
                    if len(samples) >= warmup_iterations
                    else None
                )
        elif (
            current_layer_key is not None and event.get("cat") == KERNEL_EVENT_CATEGORY
        ):
            found_kernel_event = True
            if len(durations_by_layer[current_layer_key]) >= warmup_iterations:
                duration = event_duration_seconds(current_layer_key, event)
                current_duration = (
                    current_duration + duration
                    if current_duration is not None and duration is not None
                    else None
                )

    append_current_sample()
    if not found_cuda_node:
        raise ValueError("GPU profile contains no CUDA node events")
    if not found_kernel_event:
        raise ValueError(
            "GPU profile contains no kernel events; ONNX Runtime must be built "
            "with --enable_cuda_profiling and CUPTI must be available"
        )

    return dict(durations_by_layer)
