"""
    RelevanceRankAccuracy

Compute the relevance rank accuracy between a feature-importance ranking and a
ground-truth segmentation mask.

The metric measures the ratio of high-intensity relevances that fall within the
ground-truth mask. With `P_top-k` being the set of pixels sorted by their
relevance in decreasing order up to the `k`-th pixel (where `k = |GT|`), the
rank accuracy is:

    rank accuracy = |P_top-k ∩ GT| / |GT|

A score of 1.0 means all top-k attributions lie within the mask; 0.0 means none
do. Higher scores are better.
"""
@kwdef struct RelevanceRankAccuracy <: AbstractLocalisationMetric
    normalize_config::NormalizationConfig = NormalizationConfig()
end

scoredirection(::RelevanceRankAccuracy) = higherisbetter

function evaluate(
        metric::RelevanceRankAccuracy,
        method::AbstractXAIMethod,
        x::AbstractArray{T, N},
        y::AbstractVector{<:Integer},
        y_pred::AbstractMatrix{<:Real},
        a::AbstractArray{T, N};
        s::Union{Nothing, AbstractArray{Bool, N}} = nothing
    ) where {T, N}

    isnothing(s) && error("RelevanceRankAccuracy requires a segmentation mask 's' to be passed.")
    @assert size(s) == size(a) "Segmentation mask must have the same size as the explanation."

    return relevance_rank_accuracy(metric, a, s)
end

function relevance_rank_accuracy(
        metric::RelevanceRankAccuracy,
        a::AbstractArray{T, N},
        s::AbstractArray{Bool, N}
    ) where {T, N}

    a_processed = normalize_explanations(a, metric.normalize_config)

    _size = size(a_processed)
    batch_size = _size[end]
    num_features = prod(_size[1:(end - 1)])

    A_flat = reshape(a_processed, num_features, batch_size)
    S_flat = reshape(s, num_features, batch_size)

    scores = Vector{T}(undef, batch_size)

    for i in 1:batch_size
        a_instance = @view A_flat[:, i]
        s_instance = @view S_flat[:, i]

        k = sum(s_instance)

        if k == 0
            scores[i] = T(NaN)
            continue
        end

        sorted_indices = sortperm(a_instance; rev=true)
        top_k_indices = @view sorted_indices[1:k]

        hits = sum(s_instance[top_k_indices])
        scores[i] = hits / k
    end

    return scores
end
