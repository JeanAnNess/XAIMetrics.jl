```@meta
CurrentModule = XAIMetrics
```

# Robustness

Robustness metrics evaluate how stable explanations are under small perturbations.

## Local Lipschitz Estimate

```@docs
LocalLipschitzEstimate
```

## Average Sensitivity

```@docs
AvgSensitivity
```

## Typical settings

- `nr_samples`: number of perturbation samples per input
- `perturb_config`: perturbation distribution and scale/range
- `normalize_config`: optional normalization before comparison
- `return_nan_when_prediction_changes`: optional masking when predictions flip
