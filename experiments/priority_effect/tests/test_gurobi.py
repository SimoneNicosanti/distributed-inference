import os
from dataclasses import replace

import pytest
from priority_effect.scenarios import Spec, generate, paired_problem
from priority_effect.solver import solve

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("PRIORITY_EFFECT_INTEGRATION") != "1",
        reason="Set PRIORITY_EFFECT_INTEGRATION=1; requires a working Gurobi license",
    ),
]


@pytest.mark.asyncio
async def test_real_joint_solves_and_fcfs_nesting(config, prepared):
    problem, _ = generate(config, prepared, Spec(42, 2, 0.3, "heterogeneous"))
    outcomes = [await solve(paired_problem(problem, p), config) for p in (1, 2)]
    for record in outcomes:
        assert record["feasible_incumbent"], record.get("error_details") or record
        assert record["solution_extracted"], record
        assert record["raw_solver_statistics"]["incumbent_history"]
        assert record["aggregate_normalized_tardiness"] < 1e-5
        assert record["objective_recomputed"] == pytest.approx(
            record["objective_value"], rel=1e-5
        )
    # The same all-class-1 reference placement must remain feasible with P=2.
    references = [
        await solve(paired_problem(problem, p), config, reference=True) for p in (1, 2)
    ]
    assert all(record["feasibility_certificate"] for record in references), references
    assert references[0]["objective_value"] == pytest.approx(
        references[1]["objective_value"], rel=0.02
    )


@pytest.mark.asyncio
async def test_native_infeasible_iis_and_timeout(config, prepared, tmp_path):
    problem, _ = generate(config, prepared, Spec(42, 2, 0.3, "heterogeneous"))
    impossible = replace(
        problem,
        flows={
            flow: replace(data, max_response_time_s=1e-12)
            for flow, data in problem.flows.items()
        },
    )
    iis_path = tmp_path / "reference.ilp"
    infeasible = await solve(impossible, config, reference=True, iis_path=iis_path)
    assert infeasible["termination"] == "proven_infeasible", infeasible
    assert iis_path.is_file()
    limited = config.model_copy(
        update={"solver": config.solver.model_copy(update={"timeout_s": 1e-9})}
    )
    timeout = await solve(problem, limited)
    assert timeout["termination"] == "timeout_without_incumbent", timeout
    assert not timeout["infeasibility_proven"]


@pytest.mark.asyncio
async def test_degenerate_toy_psd_failure_is_not_infeasibility(config, prepared):
    # Gurobi may reject the degenerate tiny ReLU P=2 model during convexity
    # recognition. Preserve the native failure; never enable NonConvex=2.
    problem, _ = generate(config, prepared, Spec(42, 2, 0.8, "heterogeneous"))
    record = await solve(paired_problem(problem, 2), config)
    if record["termination"] == "solver_error":
        assert "positive semi-definite" in record["error_details"]
        assert not record["infeasibility_proven"]
    else:
        assert record["feasible_incumbent"]
