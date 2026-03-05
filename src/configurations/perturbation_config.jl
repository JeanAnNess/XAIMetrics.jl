struct PerturbationConfig{F<:Function, D<:Sampleable}
    perturb_func::F
    distribution::D

    function PerturbationConfig(perturb_func::F, distribution::D) where {F<:Function, D<:Sampleable}
        new{F, D}(perturb_func, distribution)
    end
end
"""
    PerturbationConfig(func)
    PerturbationConfig(func, distribution)
    PerturbationConfig(func, params::NamedTuple)

Creates a configuration for perturbation logic.

# Examples
```julia
using XAIMetrics
using Distributions

x = rand(10)
x_gaussian = similar(x)
x_uniform = similar(x)

# Setup configurations
cfg_gaussian = PerturbationConfig(gaussian_perturbation!, (; std=0.1))
cfg_uniform = PerturbationConfig(uniform_noise!, (; lower=5))

# Apply perturbations
perturb_input!(x_gaussian, x, cfg_gaussian)
perturb_input!(x_uniform, x, cfg_uniform)
```
"""
# gaussian
PerturbationConfig(::typeof(gaussian_perturbation!)) =
    PerturbationConfig(gaussian_perturbation!, Normal(0.0, 0.1))
    
PerturbationConfig(::typeof(gaussian_perturbation!), params::NamedTuple) =
    PerturbationConfig(gaussian_perturbation!, Normal(get(params, :mean, 0.0), params.std))

#uniform
PerturbationConfig(::typeof(uniform_noise!)) =
    PerturbationConfig(uniform_noise!, Uniform(-0.02, 0.02))

PerturbationConfig(::typeof(uniform_noise!), params::NamedTuple) =
    PerturbationConfig(uniform_noise!, haskey(params, :upper) ? Uniform(params.lower, params.upper) : Uniform(-params.lower, params.lower))

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
 