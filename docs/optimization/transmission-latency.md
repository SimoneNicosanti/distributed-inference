# Source-based transmission latency

Both Gurobi assignment and cumulative representations now use
\(\xi^f_{k,s}\), independently of DNN model selection and cut volume.
The link coefficient \(\xi_{st}\) is one-way latency:
`round_trip_time_s / 2`.

For every flow, cut and source:

\[
\begin{aligned}
\xi^f_{k,s} &\ge \xi_{st}(e^f_{k,s}+e^f_{k+1,t}-1)
&&\forall t\ne s \text{ with a profiled link},\\
\xi^f_{k,s} &\le \hat{\xi}^f_s e^f_{k,s},\\
\xi^f_{k,s} &\le \sum_{t\ne s}\xi_{st}e^f_{k+1,t},\\
\xi^f_{k,s} &\ge 0.
\end{aligned}
\]

Here \(\hat{\xi}^f_s=\max_{t\ne s}\xi_{st}\), restricted to profiled outgoing
links within the flow's worker set. Its value is zero when there are no such links.
Unprofiled directed transitions are forbidden by
\(e^f_{k,s}+e^f_{k+1,t}\le1\). Existing consecutive-worker restrictions remain.

For integral placements the bounds force the selected source's latency to equal
the selected link coefficient, and every other source to zero. If the next
component is inactive, the destination upper bound forces zero even when the
current component is active. An active zero-byte cut still incurs latency.

The component latency is the affine sum \(\xi^f_k=\sum_s\xi^f_{k,s}\).
Flow response includes \(\sum_k\xi^f_k\) exactly once, outside all model sums.
The solution translator reads that same sum. No secondary objective is required.

`NetworkVariables.source_latency_s` replaces `adjacent_component_workers`.
All `q[f,k,s,t]` variables and incoming/outgoing marginal rows are removed.
Connectivity uses forbidden-pair constraints. Nominal transmission times retain
their existing placement-conditioned epigraphs.

This uses \(\sum_f(K-1)|S^f|\) latency variables. The continuous relaxation changes;
no runtime improvement is claimed without a new benchmark.

Changed production blocks, under
`services/optimizer/src/optimizer/adapters/outbound/gurobi/`:

- Network variable definitions and creation.
- Network latency and connectivity constraints.
- Response-time latency sum.
- Compact solution translator.

Public request/solution contracts are unchanged.

Validation: `test/optimizer/test_transmission_latency.py` tests both
representations, every permitted path of one/two/three active components on three
workers, and both candidate models. Minimizing and maximizing latency gives the
exact direct-product value. Cases cover inactive cuts, zero payload,
zero/asymmetric latency, missing/reversed links, a local-only flow with no links,
and exact translated latency for constant/accuracy objectives.

The new tests replace the obsolete transition-marginal comparison tests.
Component-priority matrix snapshots reflect this deliberate reformulation.
Focused verification: 145 tests passed; mypy passed on seven relevant files;
Ruff passed. The complete repository suite and a performance campaign were not run.
