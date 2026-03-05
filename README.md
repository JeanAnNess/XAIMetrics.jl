# XAIMetrics.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JeanAnNess.github.io/XAIMetrics.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JeanAnNess.github.io/XAIMetrics.jl/dev/)
[![Build Status](https://github.com/JeanAnNess/XAIMetrics.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/JeanAnNess/XAIMetrics.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/JeanAnNess/XAIMetrics.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JeanAnNess/XAIMetrics.jl)

Evaluation metrics for explanations in Julia.

`XAIMetrics.jl` focuses on quantifying explanation quality and robustness for methods that follow the Julia-XAI interface. It is designed to work with analyzers from [`ExplainableAI.jl`](https://github.com/Julia-XAI/ExplainableAI.jl) and models used in the Flux ecosystem.

## Installation

This package supports Julia `≥ 1.10`.

```julia-repl
julia> ] add XAIMetrics # The package is not yet registered so this wont work
```

For now, install from GitHub:

```julia-repl
julia> ] add https://github.com/JeanAnNess/XAIMetrics.jl
```

## Example

Compute average sensitivity scores for a batch of inputs:

```julia
using XAIMetrics
using ExplainableAI
using Flux, Metalhead

# Model and analyzer
model = ResNet(18; pretrain = true)
analyzer = InputTimesGradient(model)

# Batch input (W, H, C, N) and labels
batch_size = 4
x_batch = rand(Float32, 224, 224, 3, batch_size)
y_batch = rand(1:1000, batch_size)

# Configure perturbation and metric
perturb_cfg = PerturbationConfig(uniform_noise!, (; lower = 0.1f0, upper = 0.2f0))
metric = AvgSensitivity(
	nr_samples = 20,
	perturb_config = perturb_cfg,
	return_nan_when_prediction_changes = false,
)

# Evaluate (returns one score per sample)
scores = evaluate(metric, analyzer, x_batch; y = y_batch)
```

You can evaluate `LocalLipschitzEstimate` in the same way:

```julia
lle = LocalLipschitzEstimate(nr_samples = 20)
lle_scores = evaluate(lle, analyzer, x_batch; y = y_batch)
```

## Implemented Metrics

Current robustness metrics:

- `LocalLipschitzEstimate`
- `AvgSensitivity`

## Roadmap

Planned additions include more metrics beyond robustness (e.g., faithfulness and complexity), expanded configuration presets, and stronger parity checks with existing XAI metric libraries.