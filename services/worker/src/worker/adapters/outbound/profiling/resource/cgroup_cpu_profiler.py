import os
from collections.abc import Callable
from pathlib import Path
from time import monotonic_ns
from typing import override

from worker.application.ports.outbound.profiling.resource.cpu_profiler import (
    CpuProfiler,
)
from worker.domain.profiling.resource_profile import CpuProfile


class CgroupCpuProfiler(CpuProfiler):
    def __init__(
        self,
        cgroup_path: Path = Path("/sys/fs/cgroup"),
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        self._cpu_max_path = cgroup_path / "cpu.max"
        self._cpu_stat_path = cgroup_path / "cpu.stat"
        self._clock_ns = clock_ns

        self._previous_usage_usec: int | None = None
        self._previous_sample_ns: int | None = None
        self._previous_capacity_cores: float | None = None

    @override
    async def profile_cpu(self) -> CpuProfile:
        logical_cpu_count = self._get_logical_cpu_count()
        capacity_cores = self._get_capacity_cores(logical_cpu_count)
        usage_usec = self._get_usage_usec()
        sample_ns = self._clock_ns()

        utilization_ratio: float | None = None
        sample_duration_s: float | None = None

        if (
            self._previous_usage_usec is not None
            and self._previous_sample_ns is not None
            and self._previous_capacity_cores == capacity_cores
        ):
            elapsed_ns = sample_ns - self._previous_sample_ns
            used_usec = usage_usec - self._previous_usage_usec

            if elapsed_ns > 0 and used_usec >= 0:
                sample_duration_s = elapsed_ns / 1_000_000_000
                utilization_ratio = min(
                    used_usec
                    / 1_000_000
                    / sample_duration_s
                    / capacity_cores,
                    1.0,
                )

        self._previous_usage_usec = usage_usec
        self._previous_sample_ns = sample_ns
        self._previous_capacity_cores = capacity_cores

        return CpuProfile(
            logical_cpu_count=logical_cpu_count,
            capacity_cores=capacity_cores,
            utilization_ratio=utilization_ratio,
            sample_duration_s=sample_duration_s,
        )

    @staticmethod
    def _get_logical_cpu_count() -> int:
        logical_cpu_count = os.process_cpu_count()
        if logical_cpu_count is None or logical_cpu_count <= 0:
            raise RuntimeError("Unable to determine the process CPU count")
        return logical_cpu_count

    def _get_capacity_cores(self, logical_cpu_count: int) -> float:
        fields = self._cpu_max_path.read_text(encoding="utf-8").split()
        if len(fields) != 2:
            raise ValueError(f"Invalid cgroup CPU limit: {self._cpu_max_path}")

        quota_text, period_text = fields
        period_usec = int(period_text)
        if period_usec <= 0:
            raise ValueError(f"Invalid cgroup CPU period: {self._cpu_max_path}")

        if quota_text == "max":
            return float(logical_cpu_count)

        quota_usec = int(quota_text)
        if quota_usec <= 0:
            raise ValueError(f"Invalid cgroup CPU quota: {self._cpu_max_path}")

        return min(float(logical_cpu_count), quota_usec / period_usec)

    def _get_usage_usec(self) -> int:
        for line in self._cpu_stat_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == "usage_usec":
                usage_usec = int(fields[1])
                if usage_usec < 0:
                    break
                return usage_usec

        raise ValueError(
            f"Missing or invalid usage_usec in cgroup CPU statistics: "
            f"{self._cpu_stat_path}"
        )
