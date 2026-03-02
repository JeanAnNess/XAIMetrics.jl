```@meta
CurrentModule = XAIMetrics
```

# Configurations

Configuration objects make metric behavior explicit and reusable.

## Normalization

Normalization controls whether and how explanation values are scaled before metric computation.

- Main config type: `NormalizationConfig`
- Typical use: max-absolute scaling for comparability across samples
- Full API and functions: [Normalizations](normalizations.md)

```@autodocs
Modules = [XAIMetrics]
Pages   = ["configurations/normalization_config.jl"]
```

## Perturbation

Perturbation defines how noisy/modified inputs are generated during robustness evaluation.

- Typical use: choose perturbation function and distribution parameters
- Function-level API: [Perturbations](perturbations.md)

```@autodocs
Modules = [XAIMetrics]
Pages   = ["configurations/perturbation_config.jl"]
```

## Similarity

Similarity defines how differences between original and perturbed explanations are measured.

- Main config type: `SimilarityConfig`
- Typical use: select numerator/denominator norms and similarity function
- Full API and helper functions: [Similarities](similarities.md)

```@autodocs
Modules = [XAIMetrics]
Pages   = ["similarity_config.jl"]
```

## Where to continue

- For concrete constructor examples, see [Getting Started](generated/getting_started.md).
- For function-level details, view [Normalizations](normalizations.md), [Perturbations](perturbations.md), and [Similarities](similarities.md).

