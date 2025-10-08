module ExplanationMetrics

using XAIBase: AbstractXAIMethod, IndexSelector, analyze
using Base: @kwdef 
using LinearAlgebra: norm

const DEFAULT_NORM_FUNC = (x, y) -> norm(x - y) # ggf defaults auslagern? 

# abstracts
include("abstracts.jl")
include("helpers/utils.jl")
# functions
include("functions/functions.jl")
# include("functions/similarity_function.jl")
# include("functions/pertub_functions.jl")

# Faithfulness Metrics
include("metrics/faithfulness/pixelflipping_struct.jl")
include("metrics/faithfulness/pixelflipping_eval.jl")

# Robustness Metrics
include("metrics/robustness/robustness.jl")

function evaluate(
    metric::M, 
    model, 
    x_batch, 
    y_batch, 
    a_batch; 
    kwargs...
) where M <: AbstractXAIEvaluationMetric
    
    # Dispatch
    return _evaluate(metric, model, x_batch, y_batch, a_batch; kwargs...)
end

# For XAIBase.jl Analyzer objects
function evaluate(
    metric::M, 
    analyzer::AbstractXAIMethod, 
    x_batch, 
    y_batch; 
    kwargs...
) where M <: AbstractXAIEvaluationMetric
    return _evaluate(metric, analyzer, x_batch, y_batch; kwargs...)
end

export evaluate 

# Abstracts
export AbstractXAIEvaluationMetric
export AbstractAxiomaticMetric, AbstractComplexityMetric, AbstractLocalisationMetric
export AbstractRandomisationMetric, AbstractFaithfulnessMetric, AbstractRobustnessMetric

# Metrics

## Axoimatic

## Complexity

## Localisation

## Randomisation

## Faithfulness
export PixelFlipping

## Robustness
export LocalLipschitzEstimate, MaxSensitivity

# Functions

## normalization
export normalize_by_max_abs

## perturbation
export gaussian_noise

## similarity
export distance_euclidean, lipschitz_constant

# Helpers
export expand_indices

end # module 