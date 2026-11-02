"""Content-based software identity, including dirty production source files."""

import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from scalability_benchmark import __version__
from scalability_benchmark.catalog import file_digest
from scalability_benchmark.randomness import fingerprint


def software_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[4]
    roots = [
        repository / "experiments/scalability/src",
        repository / "experiments/scalability/scripts",
    ]
    roots.extend(
        repository / "services" / name / "src"
        for name in ("controller", "model_manager")
    )
    roots.extend(
        repository / "packages" / name / "src"
        for name in ("shared", "integration", "artifacts", "utils")
    )
    hashes = {
        path.relative_to(repository).as_posix(): file_digest(path)
        for root in roots
        for path in sorted(root.rglob("*.py"))
    }
    lock = repository / "experiments/scalability/uv.lock"
    if lock.is_file():
        hashes[lock.relative_to(repository).as_posix()] = file_digest(lock)
    versions: dict[str, str] = {}
    for package in (
        "gurobipy",
        "onnx",
        "onnx-tool",
        "onnxruntime",
        "numpy",
        "networkx",
        "pydantic",
        "pyyaml",
        "matplotlib",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        commit = "unknown"
    return {
        "experiment_version": __version__,
        "source_hash": fingerprint(hashes),
        "source_files": hashes,
        "commit": commit,
        "versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }
