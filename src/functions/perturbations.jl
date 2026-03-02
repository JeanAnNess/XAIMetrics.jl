# Perturbation functions
"""
    gaussian_perturbation!(x_perturbed, x_batch, distribution)

Sample additive Gaussian-like noise from `distribution` into `x_perturbed`
and return `x_batch .+ noise` in place.
"""
function gaussian_perturbation!(x_perturbed, x_batch, distribution::Sampleable)
    rand!(distribution, x_perturbed)
    return x_perturbed .= x_batch .+ x_perturbed
end

"""
    uniform_noise!(x_perturbed, x_batch, distribution)

Sample additive uniform-like noise from `distribution` into `x_perturbed`
and return `x_batch .+ noise` in place.
"""
function uniform_noise!(x_perturbed, x_batch, distribution::Sampleable)
    rand!(distribution, x_perturbed)
    return x_perturbed .= x_batch .+ x_perturbed
end