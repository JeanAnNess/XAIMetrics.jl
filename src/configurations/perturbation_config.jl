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


# Perturbation functions
# TODO: mit Distributions.jl
function gaussian_perturbation!(x_perturbed, x_batch; std)
    randn!(x_perturbed)    # fill with N(0,1)
    return x_perturbed .= x_batch .+ std .* x_perturbed
end

function uniform_noise!(x_perturbed, x_batch; lower, upper = nothing) # via multiple dispatch, kein nothing!
    return if upper === nothing
        # symmetric range: [-lower, lower]
        x_perturbed .= x_batch .+ (-lower .+ (2 * lower) * rand(size(x_batch)...))
    else
        x_perturbed .= x_batch .+ (lower .+ (upper - lower) * rand(size(x_batch)...))
    end
end

# Default constructors
PerturbationConfig(::typeof(gaussian_perturbation!)) =
    PerturbationConfig(gaussian_perturbation!, (; std = 0.1))

PerturbationConfig(::typeof(uniform_noise!)) =
    PerturbationConfig(uniform_noise!, (; lower = 0.02, upper = nothing))
