"""
    RelevanceRankAccuracy

Compute the relevance rank accuracy between two feature-importance rankings.

This metric evaluates how well an explanation method orders features by
importance compared to a reference ranking.
"""
@kwdef struct RelevanceRankAccuracy{FS, FN, FD} <: AbstractLocalisationMetric
    normalize_config::NormalizationConfig = NormalizationConfig()
end

scoredirection(::AvgSensitivity) = higherisbetter

"""
    evaluate(...)

Internal function to compute the AvgSensitivity for a batch of data.
"""
function evaluate(
        metric::AvgSensitivity,
        method::AbstractXAIMethod,
        x::AbstractArray{T, N},
        y::AbstractVector{<:Integer}, # true labels
        y_out::AbstractMatrix{<:Real}, # predicted logits
        a::AbstractArray{T, N};
        s::Union{Nothing, AbstractArray{Bool, N}} = nothing
    ) where {T, N}

    isnothing(s) && error("RelevanceRankAccuracy requires a segmentation mask 's' to be passed via the corresponding keyword argument.")
    @assert size(s) == size(a) "Segmentation mask must have the same size as the explanation."

    return relevance_rank_accuracy(metric, a, s)
end

"""
	relevance_rank_accuracy

Computes the rank accuracy score
"""
function relevance_rank_accuracy(
        metric::RelevanceRankAccuracy,
        a::AbstractArray{T, N},
        s::AbstractArray{Bool, N}
    ) where {T, N}

    a_processed = normalize_explanations(a, metric.normalize_config)

    _size = size(a_processed)
    batch_size = _size[end]
    num_features = prod(_size[1:(end - 1)])

    # Flatten for similarity computation (features × batch)
    A_flat = reshape(a_processed, num_features, batch_size)
    S_flat = reshape(s, num_features, batch_size)

    scores = Vector{T}(undef, batch_size)

    for i in 1:batch_size # would like to use eachcol, but since I access both A and S, wont be possible
        a_instance = @view A_flat[:, i] # attribution
        s_instance = @view S_flat[:, i] # "ground truth"

        k = sum(s_instance)

        if k == 0 # empty segmentation mask
            scores[i] = T(NaN)
            continue
        end

        # indices of desc sorted attribution -> top-k indices
        sorted_indices = sortperm(a_instance; rev = true)
        top_k_indices = @view sorted_indices[1:k]

        # amount in intersection: |top-k & s_instance|
        hits = sum(s_instance[top_k_indices])

        scores[i] = hits / k
    end

    return scores
end 
