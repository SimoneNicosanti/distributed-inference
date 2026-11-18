"""Run both representations repeatedly on exactly the same generated instances."""

import argparse
import asyncio
import itertools
from dataclasses import replace
from pathlib import Path

from scalability_benchmark.campaigns import campaign_specs
from scalability_benchmark.configuration import load_configuration
from scalability_benchmark.instances import build_scenario
from scalability_benchmark.persistence import ResultStore, atomic_json
from scalability_benchmark.provenance import software_provenance
from scalability_benchmark.randomness import fingerprint
from scalability_benchmark.solver import execute_solver

from optimizer.domain.formulation.compact.misocp.compact_formulation_kind import (
    CompactFormulationKind,
)


async def compare(
    config_path: Path, output: Path, limit: int, repetitions: int
) -> None:
    config = load_configuration(config_path)
    software = software_provenance()
    with ResultStore(output, resume=False) as store:
        specs = itertools.islice(campaign_specs(config, "main"), limit)
        for seed, group in itertools.groupby(specs, key=lambda spec: spec.seed):
            source_hash = str(software["source_hash"])
            scenario = await build_scenario(
                config, seed, output / "profiles", source_hash
            )
            manifest: dict[str, object] = {
                "scenario": scenario.as_record(),
                "software": software,
            }
            scenario_id = fingerprint(manifest)
            atomic_json(output / "scenarios" / f"{scenario_id}.json", manifest)
            for spec in group:
                for repetition in range(repetitions):
                    # Alternate order; every call still owns fresh native resources.
                    order = list(CompactFormulationKind)
                    if repetition % 2:
                        order.reverse()
                    for kind in order:
                        selected = replace(spec, formulation=kind)
                        identity = {
                            "scenario_id": scenario_id,
                            **selected.as_record(),
                            "repetition": repetition,
                        }
                        run_id = fingerprint(identity)
                        base = {"run_id": run_id, **identity}
                        store.append({**base, "state": "started"})
                        result = await execute_solver(scenario, selected, config)
                        store.append({**base, **result, "state": "finished"})
                        print(
                            f"{kind.value} repeat={repetition} status={result['termination']} runtime={result['solver_runtime_s']} nonzeros={result['original_linear_nonzeros']}",
                            flush=True,
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/default.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.limit < 1 or args.repetitions < 1:
        parser.error("--limit and --repetitions must be positive")
    asyncio.run(compare(args.config, args.output, args.limit, args.repetitions))


if __name__ == "__main__":
    main()
