"""Independent commands for pilot, main, sensitivity, planning and analysis."""

import argparse
import asyncio
import itertools
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from optimizer.domain.problem.compact.compact_optimization_problem import (
    ObjectiveKind,
)
from scalability_benchmark.campaigns import CampaignName, campaign_specs
from scalability_benchmark.configuration import (
    BENCHMARK_OBJECTIVES,
    ExperimentConfig,
    load_configuration,
)
from scalability_benchmark.instances import build_scenario, request_record
from scalability_benchmark.persistence import ResultStore, atomic_json
from scalability_benchmark.provenance import software_provenance
from scalability_benchmark.randomness import fingerprint
from scalability_benchmark.solver import execute_solver


async def run_campaign(
    config: ExperimentConfig,
    campaign: CampaignName,
    output: Path,
    *,
    resume: bool,
    retry_errors: bool,
    limit: int | None,
    objectives: tuple[ObjectiveKind, ...] | None = None,
) -> None:
    provenance = software_provenance()
    provenance_hash = fingerprint(provenance)
    specs = campaign_specs(config, campaign, objectives=objectives)
    if limit is not None:
        specs = iter(itertools.islice(specs, limit))
    with ResultStore(output, resume=resume, retry_errors=retry_errors) as store:
        for seed, seed_specs in itertools.groupby(specs, key=lambda item: item.seed):
            source_hash = provenance["source_hash"]
            assert isinstance(source_hash, str)
            profile_software_hash = fingerprint(
                {"source_hash": source_hash, "versions": provenance["versions"]}
            )
            scenario = await build_scenario(
                config, seed, output / "profiles", profile_software_hash
            )
            manifest: dict[str, object] = {
                "scenario": scenario.as_record(),
                "software": provenance,
            }
            scenario_id = fingerprint(manifest)
            atomic_json(output / "scenarios" / f"{scenario_id}.json", manifest)
            metadata_pending = True
            for spec in seed_specs:
                run_id = fingerprint(
                    {
                        "scenario_id": scenario_id,
                        "specification": spec.as_record(),
                        "software": provenance_hash,
                    }
                )
                if store.completed(run_id):
                    continue
                base = {
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    **spec.as_record(),
                    "model_names": [
                        model.entry.settings.name
                        for model in scenario.models[: spec.model_count]
                    ],
                    "flows": [
                        request_record(request)
                        for request in scenario.flows[spec.load][: spec.flow_count]
                    ],
                    "solver_gap_target": config.solver.relative_gap,
                    "scenario_manifest": f"scenarios/{scenario_id}.json",
                }
                started: dict[str, object] = {
                    **base,
                    "state": "started",
                    "started_at": datetime.now(UTC).isoformat(),
                }
                if metadata_pending:
                    started["scenario_metadata"] = manifest
                    metadata_pending = False
                store.append(started)
                result = await execute_solver(scenario, spec, config)
                store.append(
                    {
                        **base,
                        **result,
                        "state": "finished",
                        "finished_at": datetime.now(UTC).isoformat(),
                    }
                )
                print(
                    f"{run_id[:12]} {spec.objective.value} {result['termination']} runtime={result['solver_runtime_s']}",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("pilot", "main", "sensitivity", "plan", "analyze")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config/default.yaml",
    )
    parser.add_argument(
        "--seed", type=int, action="append", help="Repeat to override the seed list"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=(*BENCHMARK_OBJECTIVES, "all"),
        help="Run each selected objective independently; all selects all four (default: YAML objective)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="Explicitly limit the selected campaign"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Generate analysis after the selected campaign",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    objectives: tuple[ObjectiveKind, ...] | None = None
    if args.objectives:
        if "all" in args.objectives:
            if len(args.objectives) != 1:
                parser.error(
                    "--objectives all cannot be combined with individual objectives"
                )
            objectives = BENCHMARK_OBJECTIVES
        else:
            objectives = tuple(
                dict.fromkeys(ObjectiveKind(value) for value in args.objectives)
            )
    config = load_configuration(args.config)
    if args.seed:
        config = ExperimentConfig.model_validate(
            {**config.model_dump(), "seeds": tuple(args.seed)}
        )
    output = args.output.resolve() if args.output else config.root / config.output
    if args.command == "analyze":
        from scalability_benchmark.analysis import analyze

        analyze(output)
    elif args.command == "plan":
        for name in ("pilot", "main", "sensitivity"):
            specs = list(campaign_specs(config, name, objectives=objectives))
            print(
                f"{name}: {len(specs)} runs; solver budget={sum(spec.timeout_s for spec in specs) / 3600:g} h"
            )
    else:
        try:
            asyncio.run(
                run_campaign(
                    config,
                    cast(CampaignName, args.command),
                    output,
                    resume=args.resume,
                    retry_errors=args.retry_errors,
                    limit=args.limit,
                    objectives=objectives,
                )
            )
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        if args.analyze:
            from scalability_benchmark.analysis import analyze

            analyze(output)


if __name__ == "__main__":
    main()
