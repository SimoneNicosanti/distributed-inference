# Compact reduced time linearizations

The execution and nominal transmission variables are epigraphs. They may
overestimate the actual costs of the selected partition and placement; equality
does not follow from an optimal status, particularly for objectives with flat
regions such as zero tardiness. This supersedes the earlier exact-linearization
policy and its exact-auxiliary-value tests.

## Execution

For each flow, candidate configuration, component and worker, retain
nonnegativity and the placement-conditioned lower bound:

```
E[k,s] = sum(tau[layer,s] * x[layer,k])
E_hat[s] = sum(tau[layer,s])
T[k,s] >= E[k,s] - E_hat[s] * (1 - e[k,s])
```

The upper bound is cumulative across all component slots, including inactive
ones, independently for each flow, configuration and worker:

```
sum(T[k,s] for k in components) <= E_hat[s] * u
```

An unselected candidate therefore contributes zero execution cost everywhere.
For a selected candidate, the total epigraph cost on each worker cannot exceed
that worker's full-model execution budget. The independent selection-gated
per-component bound was reverted; the cumulative bound is the current policy.
The previous exactness bounds
`T <= E_hat * e` and `T <= E` are removed. The individual constant variable upper
bound, unsupported-layer constraints and minimum-cost valid inequalities remain.
Only available and permitted profiles contribute to `E_hat`; relay-only local
workers keep boundary profiles only.

## Transmission

Retain nonnegative variables, the lower bound conditioned on the source and
destination pair, `T <= T_hat * e_source`, and `T <= delta / beta_min`.
`beta_min` considers reachable, eligible outgoing links; sources without such
links have zero transmission time. The destination-conditioned upper bound is
removed. The existing constant upper bound, minimum-time inequality and
transition constraints remain. Empty cuts, unselected candidates and unplaced
sources still contribute zero nominal transmission.

## Translation and validation

`ComponentSolution.mean_execution_time_s` is recomputed from the selected
configuration's layer profiles on its actual worker, rather than copied from
the execution epigraph. The returned flow response sums batch fill and returned
component costs. Queue waits and transmission responses retain the existing
solver estimates; transmission response includes the PS queue approximation
and is not merely nominal `delta / beta`.

The native objective, bound, gap and termination retain their solver meanings.
The returned flow response can consequently be smaller than its auxiliary
model value. No objective or queue approximation is changed by this migration.
`MisocpCompactFormulation` already supplies the necessary coefficients and its
contract is unchanged.

`test/optimizer/test_compact_reduced_times.py` checks cumulative budgets,
unselected candidates, permitted execution/transmission slack, prohibited
underestimates and zero cases, plus native full-model translation under all five
objectives with deliberately inflated execution variables. Previous tests
requiring exact auxiliary values remain unchanged and predate this policy.
`test/optimizer/test_compact_individual_execution_bounds.py` describes the
reverted independent-bound policy and is left unchanged as stale coverage.
