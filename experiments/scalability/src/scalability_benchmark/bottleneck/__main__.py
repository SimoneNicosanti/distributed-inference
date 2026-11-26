"""python -m scalability_benchmark.bottleneck {plan,run,analyze}."""

import argparse
import asyncio
import json
from pathlib import Path

from scalability_benchmark.bottleneck.configuration import cases, load_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "analyze"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/scalability/bottleneck/config/default.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-instances", type=int)
    args = parser.parse_args()
    if args.max_instances is not None and args.max_instances < 1:
        parser.error("--max-instances must be positive")
    study, base = load_study(args.config)
    if args.output:
        study = study.model_copy(update={"output": args.output.resolve()})
    if args.command == "plan":
        selected = cases(study)
        if args.max_instances is not None:
            selected = selected[: args.max_instances]
        solves = len(selected) * len(study.ablations) * len(study.start_modes)
        extra_b_solves = 0 if "cold" in study.start_modes else len(selected)
        print(
            json.dumps(
                {
                    "instances": len(selected),
                    "measured_solves": solves,
                    "native_budget_upper_bound_s": solves
                    * study.solver.time_limit_seconds,
                    "reference_budget_upper_bound_s": len(selected)
                    * study.reference.validation_seconds
                    + extra_b_solves * study.solver.time_limit_seconds,
                    "reference_source": "cold B with predefined variants",
                    "reference_reuses_measured_b": "cold" in study.start_modes,
                    "extra_reference_b_solves": extra_b_solves,
                    "presolve_probe_budget_upper_bound_s": (solves + extra_b_solves)
                    * min(
                        study.solver.presolve_probe_seconds,
                        study.solver.time_limit_seconds,
                    )
                    if study.solver.measure_presolve
                    else 0,
                    "output": str(study.output),
                    "cases": [case.record() for case in selected],
                },
                indent=2,
            )
        )
    elif args.command == "run":
        from scalability_benchmark.bottleneck.runner import run

        asyncio.run(
            run(study, base, resume=args.resume, max_instances=args.max_instances)
        )
        from scalability_benchmark.bottleneck.analysis import analyze

        analyze(study.output)
    else:
        from scalability_benchmark.bottleneck.analysis import analyze

        analyze(study.output)


if __name__ == "__main__":
    main()
