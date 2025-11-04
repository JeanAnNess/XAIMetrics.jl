module XAIMetrics # Rename XAIMetrics

using XAIBase: AbstractXAIMethod, IndexSelector, analyze
using Base: @kwdef 
using LinearAlgebra

using Random
using Statistics

const DEFAULT_NORM_FUNC = (x, y) -> norm(x - y) # ggf defaults auslagern? 

# abstracts
include("abstracts.jl")

# configurations
include("configurations/normalization_config.jl")
include("configurations/perturbation_config.jl")
include("helpers/utils.jl")

# functions
include("functions/normalize_functions.jl")
include("functions/similarity_function.jl")

# Faithfulness Metrics
#include("metrics/faithfulness/pixelflipping_struct.jl")
#include("metrics/faithfulness/pixelflipping_eval.jl")

# Robustness Metrics
include("metrics/robustness/locallipschitz.jl")

abstract type ScoreDirection end
struct LowerIsBetter <: ScoreDirection end
struct HigherIsBetter <: ScoreDirection end

const lowerisbetter = LowerIsBetter()
const higherisbetter = HigherIsBetter()


"""
    evaluate(metric, method, x; y, s)
"""
function evaluate(
    metric::AbstractXAIMetric,
    method::AbstractXAIMethod,
    x::AbstractArray{T, N};
    y::AbstractVector{<:Integer},
    s::Union{Nothing, AbstractArray{Bool, N}} = nothing,
    kwargs...
) where {T, N}

    if N < 2
        error("Input x must have at least 2 dimensions (features and batch dimension).")
    end

    expl = analyze(x, method)
    y_out = expl.output
    a = expl.val

    # make y into optional, also in lle
    return evaluate(metric, method, x, y, y_out, a; s=s, kwargs...)
end


export evaluate 

# Abstracts
export AbstractXAIMetric
export AbstractAxiomaticMetric, AbstractComplexityMetric, AbstractLocalisationMetric
export AbstractRandomisationMetric, AbstractFaithfulnessMetric, AbstractRobustnessMetric

# Configurations
export NormalizationConfig, PerturbationConfig

# Metrics

## Axoimatic

## Complexity

## Localisation

## Randomisation

## Faithfulness
#export PixelFlipping

## Robustness
export LocalLipschitzEstimate

# Functions

## normalization
export normalize_by_max_abs

## perturbation
export gaussian_perturbation!, uniform_noise!, perturb_input!

## similarity
export distance_euclidean, distance_manhattan, lipschitz_constant

# Helpers
export expand_indices

end # module 