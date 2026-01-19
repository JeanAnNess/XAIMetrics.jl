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

# Functions

"""
    distance_euclidean(a::AbstractArray, b::AbstractArray)

Calculate Euclidean distance between two arrays (e.g., images or explanations).
"""
function distance_euclidean(a::AbstractArray, b::AbstractArray)
    return norm(a-b)
end


function distance_manhattan(a::AbstractArray, b::AbstractArray)
    return sum(abs.(a .- b))
end

"""
    lipschitz_constant(a, b, c, d; norm_numerator, norm_denominator)

Compute the batched local Lipschitz constant for a set of explanations and inputs.
Flatted matrices.
TODO: write docstrings
"""
function lipschitz_constant(
        a::AbstractMatrix, # aatribution
        b::AbstractMatrix, # attribution perturbt
        c::AbstractMatrix, # input
        d::AbstractMatrix; # input perturbt
        norm_numerator = distance_manhattan,
        norm_denominator = distance_euclidean
    )

    # alle matritzen sollen 1-indexing sein. wenn nicht -> error werfen.  Base.oneindexing.  schauen in Common mistakes
    eps = eps(eltype(a))

    num_samples = size(a, 2)
    scores = zeros(eltype(a), num_samples)

    for i in 1:num_samples
        a_col = @view a[:, i]
        b_col = @view b[:, i]
        c_col = @view c[:, i]
        d_col = @view d[:, i]

        numerator = norm_numerator(a_col, b_col)
        denominator = norm_denominator(c_col, d_col)
        scores[i] = numerator / (denominator + eps)
    end

    return scores
end

"""
    difference(a, b)

Calculate the difference between two images or explanations.
"""
function difference(a::AbstractMatrix, b::AbstractMatrix)
    return a-b
end

"""
    sensitivity_ratio(A_orig, A_perturbed, X_orig, X_perturbed; norm_numerator, norm_denominator, kwargs...)

Computes the ratio `norm_numerator(A_orig - A_perturbed) / norm_denominator(A_orig)`.
"""
function sensitivity_ratio(
    A_orig::AbstractMatrix{T}, A_perturbed::AbstractMatrix{T},
    X_orig::AbstractMatrix{T}, X_perturbed::AbstractMatrix{T}; # X args are unused
    norm_numerator, norm_denominator,
    kwargs... # Accept extra keywords
) where T
    sensitivities = A_orig - A_perturbed
    
    numerator = norm_numerator(sensitivities)     # Shape (1, batch_size)
    denominator = norm_denominator(A_orig)       # Shape (1, batch_size)
    
    # Handle division by zero
    ratio = numerator ./ denominator
    
    # Set to NaN if denominator was 0
    ratio[denominator .== 0] .= T(NaN)

    return vec(ratio) # Shape (batch_size,)
end