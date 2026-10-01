import os
from pathlib import Path
from typing import override

from worker.application.ports.outbound.profiling.resource.ram_profiler import (
    RamProfiler,
)
from worker.domain.profiling.resource_profile import RamProfile


class CgroupRamProfiler(RamProfiler):
    def __init__(
        self,
        cgroup_path: Path = Path("/sys/fs/cgroup"),
        proc_meminfo_path: Path = Path("/proc/meminfo"),
        proc_self_statm_path: Path = Path("/proc/self/statm"),
        page_size_bytes: int | None = None,
    ) -> None:
        self._memory_max_path = cgroup_path / "memory.max"
        self._memory_current_path = cgroup_path / "memory.current"
        self._proc_meminfo_path = proc_meminfo_path
        self._proc_self_statm_path = proc_self_statm_path

        resolved_page_size = (
            os.sysconf("SC_PAGE_SIZE")
            if page_size_bytes is None
            else page_size_bytes
        )
        if resolved_page_size <= 0:
            raise ValueError("The system page size must be positive")
        self._page_size_bytes = resolved_page_size

    @override
    async def profile_ram(self) -> RamProfile:
        memory_limit = self._memory_max_path.read_text(encoding="utf-8").strip()

        if memory_limit == "max":
            capacity_bytes, available_bytes = self._get_host_memory()
            used_bytes = capacity_bytes - available_bytes
        else:
            capacity_bytes = self._parse_non_negative_integer(
                memory_limit,
                self._memory_max_path,
            )
            if capacity_bytes == 0:
                raise ValueError(
                    f"Invalid cgroup memory capacity: {self._memory_max_path}"
                )

            used_bytes = self._read_non_negative_integer(self._memory_current_path)
            available_bytes = max(capacity_bytes - used_bytes, 0)

        return RamProfile(
            capacity_bytes=capacity_bytes,
            used_bytes=used_bytes,
            available_bytes=available_bytes,
            worker_rss_bytes=self._get_worker_rss_bytes(),
        )

    def _get_host_memory(self) -> tuple[int, int]:
        values: dict[str, int] = {}

        for line in self._proc_meminfo_path.read_text(encoding="utf-8").splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator or key not in {"MemTotal", "MemAvailable"}:
                continue

            fields = raw_value.split()
            if len(fields) != 2 or fields[1] != "kB":
                raise ValueError(f"Invalid memory information: {self._proc_meminfo_path}")

            values[key] = self._parse_non_negative_integer(
                fields[0],
                self._proc_meminfo_path,
            ) * 1024

        if "MemTotal" not in values or "MemAvailable" not in values:
            raise ValueError(
                f"Missing memory information in: {self._proc_meminfo_path}"
            )

        capacity_bytes = values["MemTotal"]
        available_bytes = values["MemAvailable"]
        if capacity_bytes == 0 or available_bytes > capacity_bytes:
            raise ValueError(f"Invalid memory information: {self._proc_meminfo_path}")

        return capacity_bytes, available_bytes

    def _get_worker_rss_bytes(self) -> int:
        fields = self._proc_self_statm_path.read_text(encoding="utf-8").split()
        if len(fields) < 2:
            raise ValueError(
                f"Invalid process memory information: {self._proc_self_statm_path}"
            )

        resident_pages = self._parse_non_negative_integer(
            fields[1],
            self._proc_self_statm_path,
        )
        return resident_pages * self._page_size_bytes

    @classmethod
    def _read_non_negative_integer(cls, path: Path) -> int:
        return cls._parse_non_negative_integer(
            path.read_text(encoding="utf-8").strip(),
            path,
        )

    @staticmethod
    def _parse_non_negative_integer(value: str, source: Path) -> int:
        try:
            parsed_value = int(value)
        except ValueError as error:
            raise ValueError(f"Invalid integer in: {source}") from error

        if parsed_value < 0:
            raise ValueError(f"Negative integer in: {source}")
        return parsed_value
