"""Sequential, resumable runs with a shared immutable problem per ablation set."""

from pathlib import Path

from scalability_benchmark.bottleneck.configuration import Study, cases
from scalability_benchmark.bottleneck.instances import instance, jsonable
from scalability_benchmark.bottleneck.model_selection import (
    REFERENCE_POLICY,
    assignment_record,
    predefined_models,
)
from scalability_benchmark.bottleneck.reference import obtain
from scalability_benchmark.bottleneck.solver import solve
from scalability_benchmark.configuration import ExperimentConfig
from scalability_benchmark.instances import build_scenario
from scalability_benchmark.persistence import ResultStore, atomic_json
from scalability_benchmark.provenance import software_provenance
from scalability_benchmark.randomness import fingerprint


async def run(
    study: Study,
    base: ExperimentConfig,
    *,
    resume: bool = False,
    max_instances: int | None = None,
) -> Path:
    planned = cases(study)
    if max_instances is not None:
        planned = planned[:max_instances]
    software = software_provenance()
    profile_software_hash = fingerprint(
        {
            "source_hash": software["source_hash"],
            "versions": software.get("versions", {}),
        }
    )
    output = study.output
    scenarios = {}
    with ResultStore(output, resume=resume) as store:
        manifest = {
            "reference_policy": REFERENCE_POLICY,
            "study": study.model_dump(mode="json"),
            "base": base.model_dump(mode="json"),
            "software": software,
            "planned_instances": len(planned),
            "planned_solves": len(planned)
            * len(study.ablations)
            * len(study.start_modes),
        }
        identity = fingerprint(manifest)
        atomic_json(output / "manifests" / f"{identity}.json", manifest)
        for index, case in enumerate(planned, 1):
            spec = case.spec
            print(
                f"instance {index}/{len(planned)} F={spec.flow_count} seed={spec.seed} objective={spec.objective.value} sweep={case.sweep}:{case.sweep_value}",
                flush=True,
            )
            if spec.seed not in scenarios:
                scenario = await build_scenario(
                    base, spec.seed, base.root / ".cache", profile_software_hash
                )
                scenarios[spec.seed] = scenario
            scenario = scenarios[spec.seed]
            problem = instance(scenario, case)
            model_choices = predefined_models(problem, spec.seed)
            model_assignment = assignment_record(model_choices)
            problem_record = jsonable(problem)
            problem_id = fingerprint(problem_record)
            scenario_id = fingerprint(
                {
                    "source": software["source_hash"],
                    "versions": software.get("versions", {}),
                    "problem": problem_id,
                    "formulation": study.formulation.value,
                    "relative_error": base.optimization.relative_error,
                }
            )
            atomic_json(
                output / "instances" / f"{scenario_id}.json",
                {
                    "scenario_id": scenario_id,
                    "problem_fingerprint": problem_id,
                    "problem": problem_record,
                    "reference_policy": REFERENCE_POLICY,
                    "model_assignment": model_assignment,
                    "requests": [
                        {
                            "flow_id": str(key.flow_id),
                            "arrival_rate_per_s": data.arrival_rate_per_s,
                            "deadline_s": data.max_response_time_s,
                            "candidates": len(data.configurations),
                        }
                        for key, data in problem.flows.items()
                    ],
                    "reference_time_s": scenario.reference_time_s,
                    "nominal_reference_load": spec.load,
                    "realized_nominal_reference_load": scenario.reference_time_s
                    * sum(f.arrival_rate_per_s for f in problem.flows.values()),
                    "case": case.record(),
                    "source_hash": software["source_hash"],
                    "model_artifacts": [
                        {"name": m.entry.settings.name, "artifacts": m.entry.artifacts}
                        for m in scenario.models[: spec.model_count]
                    ],
                    "workers": [
                        w.as_record()
                        for w in scenario.workers
                        if w.id in problem.workers
                    ],
                },
            )
            # Identity includes settings so a different acquisition policy cannot reuse a failed report.
            reference_identity = fingerprint(
                {
                    "scenario_id": scenario_id,
                    "reference_policy": REFERENCE_POLICY,
                    "model_assignment": model_assignment,
                    "reference": study.reference.model_dump(),
                    "solver": study.solver.model_dump(),
                }
            )
            reference, reference_report = await obtain(
                problem,
                study,
                base.optimization.relative_error,
                output / "references",
                reference_identity,
                model_choices=model_choices,
            )
            schedule = [
                (mode, ablation)
                for mode in study.start_modes
                for ablation in study.ablations
                if (mode, ablation) != ("cold", "B")
            ]
            if "cold" in study.start_modes:
                schedule.insert(0, ("cold", "B"))
            for mode, ablation in schedule:
                run_id = fingerprint(
                    {
                        "manifest_id": identity,
                        "case": case.record(),
                        "ablation": ablation,
                        "start_mode": mode,
                        "reference_id": reference_report["reference_id"],
                    }
                )
                if store.completed(run_id):
                    continue
                metadata = {
                    **case.record(),
                    "run_id": run_id,
                    "scenario_id": scenario_id,
                    "problem_fingerprint": problem_id,
                    "manifest_id": identity,
                    "source_hash": software["source_hash"],
                    "reference_policy": REFERENCE_POLICY,
                    "model_assignment": model_assignment,
                    "reference_source_ablation": "B",
                    "reference_source_runtime_s": reference_report["acquisition"].get(
                        "solver_runtime_s"
                    ),
                    "reference_id": reference_report["reference_id"],
                    "reference_report": str(
                        output / "references" / f"{reference_identity}-report.json"
                    ),
                    "reference_preparation_wall_s": reference_report["wall_time_s"],
                    "reference_acquisition_wall_s": (
                        reference_report.get("acquisition") or {}
                    ).get("wall_time_s"),
                    "reference_validation_wall_s": (
                        reference_report.get("validation") or {}
                    ).get("wall_time_s"),
                    "ablation": ablation,
                    "start_mode": mode,
                    "state": "started",
                    "nominal_load_policy": "full_pool_reference_utilization_per_F",
                    "relative_error": base.optimization.relative_error,
                    "deadline_policy": "soft_tardiness"
                    if spec.objective.value
                    in ("arrival_weighted_normalized_delay", "deadline_violations")
                    else "original_hard_deadlines",
                }
                store.append(metadata)
                if (mode, ablation) == ("cold", "B"):
                    result = {
                        **reference_report["acquisition"],
                        "available": True,
                        "reused_reference_source_solve": True,
                    }
                elif reference is None and (ablation != "A" or mode == "warm"):
                    result = {
                        "termination": "reference_unavailable",
                        "available": False,
                        "feasible_incumbent": False,
                        "solved_to_tolerance": False,
                        "timeout_reached": False,
                        "runtime_censored": False,
                    }
                else:
                    result, _ = await solve(
                        problem,
                        study,
                        base.optimization.relative_error,
                        output / "logs" / f"{run_id}.log",
                        ablation=ablation,
                        reference=reference,
                        model_choices=model_choices,
                        warm=mode == "warm",
                    )
                    result["available"] = True
                store.append({**metadata, **result, "state": "finished"})
                print(
                    f"  {mode} {ablation}: {result.get('termination')} runtime={result.get('solver_runtime_s')} gap={result.get('final_gap')}",
                    flush=True,
                )
    return output
