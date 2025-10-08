using Base: @kwdef
using LinearAlgebra

"""
    LocalLipschitzEstimate(; nr_samples=200, similarity_func, norm_numerator, norm_denominator, perturb_std=0.1, return_nan_when_prediction_changes=false)
    <: AbstractRobustnessMetric

Robustness metric that tests the consistency in the explanation for neighboring examples
by estimating the local Lipschitz constant.

## Keyword arguments
- `nr_samples::Int`: The number of Monte Carlo samples iterated
- `similarity_func::Function`: Function to compute the Lipschitz ratio.
- `norm_numerator::Function`: Function for norm calculations on the numerator (distance between explanations).
- `norm_denominator::Function`: Function for norm calculations on the denominator (distance between inputs).
- `perturb_std::Float64`: The standard deviation of the Gaussian noise added for perturbation (default: 0.1).
- `return_nan_when_prediction_changes::Bool`: If true, samples where the model's prediction changes after perturbation are ignored (result is NaN for that sample).

TODO: add more perturb functions, add normalization functions
"""
@kwdef struct LocalLipschitzEstimate{FS, FN, FD} <: AbstractRobustnessMetric
    nr_samples::Int = 200
    similarity_func::FS = lipschitz_constant
    norm_numerator::FN = DEFAULT_NORM_FUNC
    norm_denominator::FD = DEFAULT_NORM_FUNC
    perturb_std::Float64 = 0.1
    return_nan_when_prediction_changes::Bool = false
    normalise::Bool = true
end