# Historical full-A-reference pilot — 2026-11-24

These measurements use the former A-derived reference protocol. The current experiment
predeclares variants and derives its reference from B cold; no new campaign has been
launched for that protocol. Do not pool these historical measurements with B-derived runs.

Complete production cumulative MISOCP; component priorities; normalized response objective (arrival-weighted mean T/D), with original hard deadlines. Nominal load L=0.8; device + one edge GPU; one thread; solver seed 0; instance seed 42; native solve budget **5 s**, relative gap 1%.

The comparable pilot uses F=5,10, three candidate models and K=5, P=2. Cold and common-reference warm starts are separate. The earlier smoke uses F=2,3, M=2, K=3 and is **not** part of the same controlled F progression.

## Measured outcomes

|F|Start|A: full|B: models fixed|C: partition fixed|D: deployment fixed|P: priorities fixed|E: all discrete fixed|
|---:|---|---:|---:|---:|---:|---:|---:|
|5|cold|timeout*|0.831 s|0.081 s|0.082 s|0.070 s|0.073 s|
|5|warm|timeout*|1.202 s|0.080 s|0.088 s|0.082 s|0.069 s|
|10|cold|timeout*|3.535 s|0.187 s|0.246 s|0.187 s|0.213 s|
|10|warm|timeout*|3.765 s|0.187 s|0.192 s|0.207 s|0.153 s|

*Timeout durations are censored observations, not exact times to optimality. All B–E/P solves reach the configured 1% tolerance. Cold A has no incumbent at either F; warm A retains a feasible incumbent, with gaps 82.66% (F=5) and 88.32% (F=10). Every returned incumbent passes the recorded strict native feasibility-quality checks.

The largest completed-pair conditional runtime reduction is B→C: approximately 10–15x at F=5 and 19–20x at F=10. A→B has no exact runtime ratio because A times out. The tested difficulty rule flags A at the earliest pilot point F=5 **under this 5 s budget**; it is not a threshold established for the default 300 s budget.

**Objective quality changes substantially at F=10:** cold B gives J=0.0632411, whereas C gives J=0.0907464 (about 43.5% higher for this minimization objective). The common feasible reference partition is not optimal. Thus the observed reduction supports a conditional diagnostic observation; it does not establish an equivalent-quality speedup or an additive cost of partitioning.

Placement/priorities/auxiliary fixing produce much smaller, sometimes negative runtime changes once the partition is fixed. E solves in 0.069–0.213 s in this pilot. This gives no evidence that the remaining continuous problem dominates these small restricted instances; it does not identify queueing equations individually or rule out growth at larger F.

## Model growth

Independent presolve-probe binary counts; not necessarily identical to optimize-time presolve:

|F|A|B|C|D|P|E|
|---:|---:|---:|---:|---:|---:|---:|
|5|20754|2481|33|33|17|0|
|10|40914|4889|51|51|17|0|

E has zero free native integer variables in every measured run. Model selection and partition fixing strongly shrink the presolved model; placement fixing leaves these binary counts unchanged in this witness. Fixed native binary counts before presolve would hide this distinction, so free-domain counts are also recorded.

## Reference budgets and validation

The first F=5/10 pilot used a 20 s reference acquisition budget. F=10 exhausted it during its root relaxation and obtained no incumbent; cold A was preserved, and the other eleven dependent runs were explicitly unavailable. Those original records remain in `results/pilot/`.

A second pilot used 60 s for acquisition, without changing instances or measured-solve tuning. The references were found after 14.889 s (F=5) and 34.288 s (F=10), then independently validated on the original equations with discrete decisions fixed. Reference preparation/construction/validation are excluded from every measured native runtime.

In these pilot snapshots, the legacy `reference_acquisition_wall_s` journal column contains total reference-preparation wall time. Acquisition and validation native/wall times are separately available in each reference report. The current runner explicitly separates preparation, acquisition and validation columns.

All comparisons within each pilot instance have identical original problem fingerprints, model signatures, reference IDs and effective native parameter dictionaries. Manifests preserve the measured source/version fingerprints. Later reporting additions expose final-objective differences; cancellation cleanup and reference timing-column naming do not alter mathematical decisions or constraints.

## Artifacts and validation

- Main pilot: [raw journal](results/pilot-reference60/runs.jsonl), [complete CSV](results/pilot-reference60/analysis/runs.csv), [automatic summary](results/pilot-reference60/analysis/summary.md), [aggregate table](results/pilot-reference60/analysis/aggregate.csv), [paired transitions](results/pilot-reference60/analysis/transitions.csv).
- Historical settings and source fingerprints are preserved in `results/pilot-reference60/manifests/`. Current YAML presets use the new B-derived protocol and therefore do not reproduce this historical reference acquisition.
- Smoke: [summary](results/smoke/analysis/summary.md). All 24 solves reached tolerance; resume kept exactly 24 run IDs while revalidating both cached references.
- Each analyzed directory has eight figure families in PNG/PDF, separately for cold/warm. Raw data, logs, references and generated figures are local ignored artifacts.
- 96 focused native/experiment regression tests, Ruff and mypy pass. See the [run instructions and file inventory](README.md).

Limits: one instance seed, two comparable F points, short solve budget, only normalized response in the empirical pilot, reference-dependent restrictions and changed best objectives. Positive tardiness, zero tardiness, native infeasibility and timeout behavior are covered by native tests. No full matrix, multi-seed statistical campaign or causal attribution is claimed.
