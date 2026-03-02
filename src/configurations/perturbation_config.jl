struct PerturbationConfig{F<:Function, D<:Sampleable}
    perturb_func::F
    distribution::D
end

# Default fallbacks
_get_default_dist(::typeof(gaussian_perturbation!)) = Normal(0.0, 0.1)
_get_default_dist(::typeof(uniform_noise!)) = Uniform(-0.02, 0.02)

_dist_from_nt(::typeof(gaussian_perturbation!), p::NamedTuple) = Normal(0.0, p.std)
_dist_from_nt(::typeof(uniform_noise!), p::NamedTuple) = haskey(p, :upper) ? Uniform(p.lower, p.upper) : Uniform(-p.lower, p.lower)
"""
    PerturbationConfig()

Creates a configuration for perturbation logic.

# Examples
```julia
using XAIMetrics
using Distributions

x = rand(10)
x_gaussian = similar(x)
x_uniform = similar(x)

# Setup configurations
cfg_gaussian = PerturbationConfig(gaussian_perturbation!; params=(; std=0.1))
cfg_uniform = PerturbationConfig(uniform_noise!; params=(; lower=5))

# Apply perturbations
perturb_input!(x_gaussian, x, cfg_gaussian)
perturb_input!(x_uniform, x, cfg_uniform)
```
"""
function PerturbationConfig(func::Function; distribution=nothing, params=nothing)
    d = if !isnothing(distribution)
        distribution
    elseif !isnothing(params)
        _dist_from_nt(func, params)
    else
        _get_default_dist(func)
    end
    return PerturbationConfig(func, d)
end

"""
    perturb_input!(x_perturbed, x_batch, config)

Apply in-place perturbation to `x_batch` using the perturbation function and
distribution stored in `config`.
"""
function perturb_input!(
        x_perturbed::AbstractArray{T},
        x_batch::AbstractArray{T},
        config::PerturbationConfig
    ) where {T}
    return config.perturb_func(x_perturbed, x_batch, config.distribution)
end

# Default constructors
PerturbationConfig(::typeof(gaussian_perturbation!)) =
    PerturbationConfig(gaussian_perturbation!, Normal(0.0, 0.1))

PerturbationConfig(::typeof(uniform_noise!)) =
    PerturbationConfig(uniform_noise!, Uniform(-0.02, 0.02))
