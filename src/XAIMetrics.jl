module XAIMetrics # Rename XAIMetrics

using XAIBase: AbstractXAIMethod, IndexSelector, analyze
using Base: @kwdef
using LinearAlgebra: norm
using Distributions: Sampleable, Normal, Uniform

using Random: AbstractRNG, rand!, default_rng
using Statistics: mean

# abstracts
include("abstracts.jl")

# functions
include("functions/norm_func.jl")
include("functions/prediction_func.jl")
include("functions/normalizations.jl")
include("functions/perturbations.jl")
include("functions/similarities.jl")

# configurations
include("configurations/normalization_config.jl")
include("configurations/perturbation_config.jl")
include("configurations/similarity_config.jl")

# Faithfulness Metrics

# Robustness Metrics
include("metrics/robustness/locallipschitz.jl")
include("metrics/robustness/avg_sensitivity.jl")

# Localisation Metrics
include("metrics/localisation/relevance_rank_accuracy.jl")

const DEFAULT_NORM_FUNC = (a, b) -> sqrt.(sum(abs2.(a .- b); dims=1))
const DEFAULT_SENS_NORM_FUNC = columnwise_l2_norm

abstract type ScoreDirection end
struct LowerIsBetter <: ScoreDirection end
struct HigherIsBetter <: ScoreDirection end

const lowerisbetter = LowerIsBetter()
const higherisbetter = HigherIsBetter()

function scoredirection(metric::AbstractXAIMetric)
    error("scoredirection not implemented for $(typeof(metric))")
end


""" 
    evaluate(metric, method, x; y, s)

XAIMetrics is backend-agnostic. Any `AbstractXAIMethod` from XAIBase.jl works,
regardless of whether it uses Zygote, Enzyme, or ForwardDiff internally.
"""
function evaluate(
        metric::AbstractXAIMetric,
        method::AbstractXAIMethod,
        x::AbstractArray{T, N};
        y::AbstractVector{<:Integer}, # batches an bildern, für jedes bild die targetclass
        s::Union{Nothing, AbstractArray{Bool, N}} = nothing,
        kwargs...
    ) where {T, N}

    if N < 2
        error("Input x must have at least 2 dimensions (features and batch dimension).")
    end

    expl = analyze(x, method, IndexSelector(y))
    y_pred = expl.output
    a = expl.val

    # make y into optional, also in lle
    return evaluate(metric, method, x, y, y_pred, a; s = s, kwargs...)
end


export evaluate, scoredirection, lowerisbetter, higherisbetter

# Abstracts
export AbstractXAIMetric
export AbstractAxiomaticMetric, AbstractComplexityMetric, AbstractLocalisationMetric
export AbstractRandomisationMetric, AbstractFaithfulnessMetric, AbstractRobustnessMetric

# Configurations
export NormalizationConfig, PerturbationConfig, SimilarityConfig

# 
export normalize_explanations

# Metrics

## Axoimatic

## Complexity

## Localisation
export RelevanceRankAccuracy

## Randomisation

## Faithfulness
#export PixelFlipping

## Robustness
export LocalLipschitzEstimate, AvgSensitivity

# Functions

# functions
export predicted_classes

## normalization
export normalize_by_max_abs, stable_division

## norm functions
export frobenius_norm, columnwise_l2_norm

## perturbation
export gaussian_perturbation!, uniform_noise!, perturb_input!

## similarity
export distance_euclidean, distance_manhattan, lipschitz_constant, difference

end # module
