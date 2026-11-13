# Compact problem: one configuration per variant

Each `CompactOptimizationProblem` declares **exactly one configuration for each
`ModelVariantId`**. The rule applies to the entire problem, including all its
flows. A declared variant without a configuration, two shape configurations of
the same variant, and configurations of undeclared variants are invalid.

This supersedes the earlier controller projection policy of exposing every
profiled shape as an admissible configuration of a variant.

`ModelConfigurationKey` still contains both the variant identifier and the full
named input-shape map. Shapes, batch size, tensor sizes, and execution costs
therefore remain explicit; they are not inferred from the variant identifier.

Several different variants may remain eligible for a flow. Each has its own
single configuration, and the solver chooses one candidate for that flow.
The rule does not limit a flow to one candidate variant and does not itself
set auxiliary costs of unselected candidates to zero.

## Enforcement and projection

The controller domain, shared HTTP DTO, and optimizer domain enforce the same
cardinality. HTTP requests violating it are rejected during request validation.

General model profiles may still describe several shapes. For the compact
projection, `CompactOptimizationProblemFactory` requires each eligible static
profile to contain exactly one shape profile. It raises
`CompactOptimizationProblemBuildError` for zero or several shape profiles;
it never silently chooses the first shape or renames shapes into variants.
The caller must resolve the intended shape before asking the factory to build
the compact problem. Execution costs must correspond to that configuration.

The solver-specific `MisocpCompactFormulationConfig` continues to contain
approximation settings. It does not choose a shape or change this invariant.

## Regression coverage

`test/optimizer/test_single_configuration_contract.py` checks both domains, the
factory, DTO round trips, and HTTP rejection before optimizer invocation.
`test/optimizer/test_single_configuration_runtime.py` runs the native full solver
with one or two variants, always with exactly one configuration per variant.

Auxiliary costs are now gated by candidate selection through the
[reduced time linearizations](compact-reduced-time-linearization.md). The
execution bound is cumulative across components for each flow, candidate and
worker. Selected-candidate variables may retain slack; translated execution
times are recomputed from the selected layers and profiles. The cardinality rule
alone does not enforce cost gating.

Earlier tests using multiple configurations of the same variant are left
unchanged and predate this contract; they must be migrated separately.
