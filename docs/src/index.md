```@meta
CurrentModule = XAIMetrics
```

# XAIMetrics.jl

`XAIMetrics.jl` provides evaluation metrics for explanation methods in Julia.
The current baseline focuses on robustness metrics and reusable configuration objects.

# Installation
To install this package and its dependencies, open the Julia REPL and run
```julia-repl
julia> ]add  XAIMetrics # not yet as the package is not registered
```

## What is available now

- Metric entry point: `evaluate`
- Robustness metrics: `LocalLipschitzEstimate`, `AvgSensitivity`
- Configurations: `NormalizationConfig`, `PerturbationConfig`
- Utility functions: normalization, perturbation, and similarity helpers

## Manual

```@contents
Pages = [
    "generated/getting_started.md"
]
Depth = 2
```

## API References

```@contents
Pages = [
    "configurations.md",
    "normalizations.md",
    "perturbations.md",
    "similarities.md"
]
Depth = 2
```

## Metrics

```@contents
Pages = [
    "metrics/robustness.md"
]
Depth = 2
```