"""
    SimilarityConfig(; similarity_func=lipschitz_constant, norm_numerator=distance_manhattan, norm_denominator=distance_euclidean, kwargs=Dict())

Configuration for similarity computation between original and perturbed explanations/inputs.
"""
@kwdef struct SimilarityConfig{FS, FN, FD}
    similarity_func::FS = lipschitz_constant
    norm_numerator::FN = distance_manhattan
    norm_denominator::FD = distance_euclidean
    kwargs::Dict = Dict()
end

"""
    compute_similarity(config, A_orig, A_perturbed, X_orig, X_perturbed)

Dispatches to the similarity function defined in the `config`.
"""
function compute_similarity(
        config::SimilarityConfig,
        A_orig::AbstractMatrix, A_perturbed::AbstractMatrix,
        X_orig::AbstractMatrix, X_perturbed::AbstractMatrix
    )
    return config.similarity_func(
        A_orig, A_perturbed,
        X_orig, X_perturbed;
        norm_numerator = config.norm_numerator,
        norm_denominator = config.norm_denominator,
        config.kwargs...
    )
end
