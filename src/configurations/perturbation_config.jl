struct PerturbationConfig
    perturb_func::Function
    params::NamedTuple
end

"""
    PerturbationConfig()

Creates a configuration for perturbation logic.

Example:
x = rand(10)
x_gaussian = similar(x)
x_uniform = similar(x)

cfg_gaussian = PerturbationConfig(gaussian_perturbation!, (; std=0.1))
cfg_uniform = PerturbationConfig(uniform_noise!, (; lower=5))

perturb_input!(x_gaussian, x, cfg_gaussian)
perturb_input!(x_uniform, x, cfg_uniform)
"""
PerturbationConfig(; perturb_func = gaussian_perturbation!, params = (; std = 0.1)) =
    PerturbationConfig(perturb_func, params)

function perturb_input!(
        x_perturbed::AbstractArray{T},
        x_batch::AbstractArray{T},
        config::PerturbationConfig
    ) where {T}
    return config.perturb_func(x_perturbed, x_batch; config.params...)
end

# Default constructors
PerturbationConfig(::typeof(gaussian_perturbation!)) =
    PerturbationConfig(gaussian_perturbation!, (; std = 0.1))

PerturbationConfig(::typeof(uniform_noise!)) =
    PerturbationConfig(uniform_noise!, (; lower = 0.02, upper = nothing))
