"""Plan, pilot, run and analyze a resumable paired priority study."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from scalability_benchmark.catalog import file_digest
from scalability_benchmark.persistence import ResultStore, atomic_json
from scalability_benchmark.provenance import software_provenance
from scalability_benchmark.randomness import fingerprint

from optimizer.domain.problem.compact import (
    compact_optimization_problem as optimizer_problem,
)
from priority_effect.analysis import analyze
from priority_effect.configuration import Config, load_config
from priority_effect.scenarios import (
    Spec,
    generate,
    jsonable,
    paired_problem,
    planned,
    prepare,
)
from priority_effect.solver import solve


def provenance() -> dict[str, Any]:
    record: dict[str, Any] = software_provenance()
    # Repository paths alone miss optimizer snapshots loaded through PYTHONPATH.
    optimizer_root = Path(optimizer_problem.__file__).resolve().parents[3]
    prefix = "services/optimizer/src/optimizer/"
    hashes = {
        key: value
        for key, value in record["source_files"].items()
        if not key.startswith(prefix)
    }
    hashes.update(
        {
            prefix + path.relative_to(optimizer_root).as_posix(): file_digest(path)
            for path in sorted(optimizer_root.rglob("*.py"))
        }
    )
    record["source_files"] = hashes
    record["optimizer_source_root"] = str(optimizer_root)
    environment = Path("gurobi.env")
    record["gurobi_env_hash"] = (
        file_digest(environment) if environment.is_file() else None
    )
    root = Path(__file__).resolve().parents[2]
    record["priority_effect_source_files"] = {
        str(path.relative_to(root)): file_digest(path)
        for path in sorted((root / "src").rglob("*.py"))
    }
    record["source_hash"] = fingerprint(
        [
            fingerprint(hashes),
            record["priority_effect_source_files"],
            record["gurobi_env_hash"],
        ]
    )
    return record


async def execute(
    config: Config,
    specs: tuple[Spec, ...],
    *,
    resume: bool,
    retry_errors: bool,
    iis: bool,
) -> None:
    software = provenance()
    prepared = {}
    with ResultStore(config.output, resume=resume, retry_errors=retry_errors) as store:
        manifest = {
            "configuration": config.model_dump(mode="json", exclude={"output"}),
            "source_hash": software["source_hash"],
        }
        manifest_path = config.output / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Output contains different configuration or source code; select a new --output"
            )
        atomic_json(manifest_path, manifest)
        atomic_json(
            config.output / f"software-{software['source_hash']}.json", software
        )
        for index, spec in enumerate(specs):
            preparation_started = perf_counter()
            if spec.seed not in prepared:
                prepared[spec.seed] = await prepare(
                    config,
                    spec.seed,
                    config.output / "profiles",
                    software["source_hash"],
                )
            problem, metadata = generate(config, prepared[spec.seed], spec)
            preparation_wall_s = perf_counter() - preparation_started
            scenario_id = fingerprint(
                [
                    metadata["problem_fingerprint"],
                    metadata["models"],
                    jsonable(spec),
                    config.model_dump(mode="json", exclude={"output"}),
                    software["source_hash"],
                ]
            )
            metadata["scenario_id"] = scenario_id
            metadata["software_hash"] = software["source_hash"]
            directory = config.output / "scenarios" / scenario_id
            atomic_json(directory / "scenario.json", metadata)
            reference_path = directory / "reference-check.json"
            if config.reference_check and (not reference_path.exists() or retry_errors):
                reference = await solve(
                    problem,
                    config,
                    reference=True,
                    iis_path=directory / "reference.ilp" if iis else None,
                )
                atomic_json(reference_path, reference)
            common = {
                name: metadata[name]
                for name in (
                    "scenario_id",
                    "seed",
                    "flow_count",
                    "load",
                    "workload",
                    "category",
                    "service_demand_cv",
                )
            }
            common.update(
                {
                    "offered_reference_load": metadata["reference"][
                        "offered_reference_load"
                    ],
                    "shared_problem_fingerprint": metadata["problem_fingerprint"],
                    "objective": config.optimization.objective.value,
                    "scenario_preparation_wall_time_s": preparation_wall_s,
                    "provenance_hash": software["source_hash"],
                }
            )
            # Alternate solve order deterministically; generation never sees P.
            order = (1, 2) if int(scenario_id[:8], 16) % 2 == 0 else (2, 1)
            strategies = [
                f"full_p{p}" for p in order if f"full_p{p}" in config.strategies
            ]
            suffixes = {
                "full_p1": "p1",
                "full_p2": "p2",
            }
            run_ids: dict[str, str] = {}

            for strategy in strategies:
                priority = 1 if strategy == "full_p1" else 2
                suffix = suffixes[strategy]
                run_id = f"{scenario_id}-{suffix}"
                run_ids[strategy] = run_id

                if store.completed(run_id):
                    continue
                record = {
                    **common,
                    "run_id": run_id,
                    "priority_count": priority,
                    "strategy": strategy,
                }
                store.append({**record, "state": "started"})
                outcome = await solve(
                    paired_problem(problem, priority),
                    config,
                    iis_path=directory / f"{suffix}.ilp" if iis else None,
                )
                outcome["wall_time_including_preparation_s"] = (
                    preparation_wall_s + outcome.get("wall_time_s", 0)
                )
                store.append({**record, **outcome, "state": "finished"})
                print(
                    json.dumps(
                        {
                            "scenario": index + 1,
                            "total": len(specs),
                            "priority_count": priority,
                            "strategy": strategy,
                            "termination": outcome["termination"],
                            "feasible_incumbent": outcome["feasible_incumbent"],
                            "mean_response_s": outcome.get(
                                "arrival_weighted_mean_response_time_s"
                            ),
                        }
                    ),
                    flush=True,
                )
            # A compact pilot report with the actual generated workload and statuses.
            report = {
                **common,
                "flows": metadata["flows"],
                "model_assignments": metadata["model_assignments"],
                "reference": metadata["reference"],
                "native_reference_check": json.loads(reference_path.read_text())
                if reference_path.exists()
                else None,
                "outcomes": {
                    strategy: store.latest[run_ids[strategy]]["termination"]
                    for strategy in strategies
                },
            }
            atomic_json(directory / "pilot-report.json", report)
    analyze(config.output, plots=False, tight_gap=config.tight_gap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "pilot", "run", "analyze"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config/default.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--iis", action="store_true", help="Export IIS for proven infeasible runs"
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="Explicitly limit scenario count for a smoke run"
    )
    parser.add_argument(
        "--timeout", type=float, help="Explicit equal per-run timeout override"
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = load_config(args.config)
    if args.output is not None:
        config = config.model_copy(update={"output": args.output.resolve()})
    if args.timeout is not None:
        config = config.model_copy(
            update={
                "solver": config.solver.model_validate(
                    {
                        **config.solver.model_dump(),
                        "timeout_s": args.timeout,
                    }
                )
            }
        )
    if args.command == "analyze":
        report = analyze(
            config.output, plots=not args.no_plots, tight_gap=config.tight_gap
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "run_count",
                        "scenario_count",
                        "valid_pair_count",
                        "feasibility",
                    )
                },
                indent=2,
            )
        )
        return
    specs = planned(config, pilot=args.command == "pilot")
    if args.limit:
        specs = specs[: args.limit]
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "scenario_count": len(specs),
                    "optimization_runs": len(config.strategies) * len(specs),
                    "strategies": config.strategies,
                    "reference_checks": len(specs) if config.reference_check else 0,
                    "scenarios": jsonable(specs),
                    "solver": config.solver.model_dump(mode="json"),
                    "priority_counts": config.priority_counts,
                    "assignment_strategy": "explicit assignment or profile-ranked ResNet/EfficientNet demand spread; generated once",
                },
                indent=2,
            )
        )
    else:
        asyncio.run(
            execute(
                config,
                specs,
                resume=args.resume,
                retry_errors=args.retry_errors,
                iis=args.iis,
            )
        )
