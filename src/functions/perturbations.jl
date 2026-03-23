function _additive_noise!(x_perturbed, x_batch, distribution::Sampleable, rng::AbstractRNG)
    rand!(rng, distribution, x_perturbed)
    return x_perturbed .= x_batch .+ x_perturbed
end

"""
    gaussian_perturbation!(x_perturbed, x_batch, distribution)
Additive Gaussian noise sampled from `distribution`.
"""
gaussian_perturbation!(x_perturbed, x_batch, d::Sampleable, rng::AbstractRNG) =
    _additive_noise!(x_perturbed, x_batch, d, rng)
"""
    uniform_noise!(x_perturbed, x_batch, distribution)
Additive uniform noise sampled from `distribution`.
"""
uniform_noise!(x_perturbed, x_batch, d::Sampleable, rng::AbstractRNG) =
    _additive_noise!(x_perturbed, x_batch, d, rng)



