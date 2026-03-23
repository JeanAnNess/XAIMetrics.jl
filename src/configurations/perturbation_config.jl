"""
    PerturbationConfig(func; rng=Random.default_rng())
    PerturbationConfig(func, distribution; rng=Random.default_rng())
    PerturbationConfig(func, params::NamedTuple; rng=Random.default_rng())

Configuration for a perturbation step in a metric evaluation pipeline.

Bundles a perturbation function, the distribution to sample noise from, and
the random number generator used for sampling. Pass an explicit `rng` for
reproducible perturbations.

# Arguments
- `func`: perturbation function with signature `func!(out, x, dist, rng)`
- `distribution`: distribution to sample additive noise from
- `params`: named tuple of distribution parameters (alternative to passing a distribution directly)
- `rng`: random number generator (default: `Random.default_rng()`)

# Examples
```julia
using XAIMetrics, Random

# Default RNG
cfg = PerturbationConfig(gaussian_perturbation!, (; std=0.1))

# Explicit RNG for reproducibility
rng = Xoshiro(42)
cfg_gaussian = PerturbationConfig(gaussian_perturbation!, (; std=0.1); rng)
cfg_uniform  = PerturbationConfig(uniform_noise!, (; lower=0.02); rng)

x = rand(Float32, 10)
x_out = similar(x)
perturb_input!(x_out, x, cfg_gaussian)
```
"""
struct PerturbationConfig{F <: Function, D <: Sampleable, R <: AbstractRNG}
    perturb_func::F
    distribution::D
    rng::R
end

# inner constructor with default rng
PerturbationConfig(f::F, d::D) where {F <: Function, D <: Sampleable} =
    PerturbationConfig(f, d, default_rng())

# gaussian
PerturbationConfig(::typeof(gaussian_perturbation!); rng::AbstractRNG=default_rng()) =
    PerturbationConfig(gaussian_perturbation!, Normal(0.0, 0.1), rng)

PerturbationConfig(::typeof(gaussian_perturbation!), params::NamedTuple; rng::AbstractRNG=default_rng()) =
    PerturbationConfig(gaussian_perturbation!, Normal(get(params, :mean, 0.0), params.std), rng)

# uniform
PerturbationConfig(::typeof(uniform_noise!); rng::AbstractRNG=default_rng()) =
    PerturbationConfig(uniform_noise!, Uniform(-0.02, 0.02), rng)

PerturbationConfig(::typeof(uniform_noise!), params::NamedTuple; rng::AbstractRNG=default_rng()) =
    PerturbationConfig(uniform_noise!,
        haskey(params, :upper) ? Uniform(params.lower, params.upper) : Uniform(-params.lower, params.lower),
        rng)

"""
    perturb_input!(x_perturbed, x_batch, config)

Apply perturbation to `x_batch` in-place, writing the result into `x_perturbed`.
Uses the perturbation function, distribution, and RNG stored in `config`.
"""
function perturb_input!(
        x_perturbed::AbstractArray{T},
        x_batch::AbstractArray{T},
        config::PerturbationConfig
    ) where {T}
    return config.perturb_func(x_perturbed, x_batch, config.distribution, config.rng)
end