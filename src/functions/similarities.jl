"""
    distance_euclidean(a::AbstractArray, b::AbstractArray)

Calculate Euclidean distance between two arrays (e.g., images or explanations).
"""
function distance_euclidean(a::AbstractMatrix, b::AbstractMatrix)
    return sqrt.(sum(abs2.(a .- b); dims=1))
end

function distance_euclidean(a::AbstractVector, b::AbstractVector)
    return norm(a - b)
end

"""
    distance_manhattan(a::AbstractArray, b::AbstractArray)

Calculate Manhattan (L1) distance between two arrays.
"""
function distance_manhattan(a::AbstractMatrix, b::AbstractMatrix)
    return sum(abs.(a .- b); dims=1)
end

function distance_manhattan(a::AbstractVector, b::AbstractVector)
    return sum(abs, a - b)
end

"""
    lipschitz_constant(a, b, c, d; norm_numerator, norm_denominator)

Compute the batched local Lipschitz constant for a set of explanations and inputs.
"""
function lipschitz_constant(
        a::AbstractMatrix,
        b::AbstractMatrix,
        c::AbstractMatrix,
        d::AbstractMatrix;
        norm_numerator = distance_manhattan,
        norm_denominator = distance_euclidean
    )
    epsilon = eps(eltype(a))
    return vec(norm_numerator(a, b) ./ (norm_denominator(c, d) .+ epsilon))
end

"""
    difference(a, b)

Calculate the difference between two images or explanations.
"""
function difference(a::AbstractMatrix, b::AbstractMatrix)
    return a - b
end

"""
    sensitivity_ratio(A_orig, A_perturbed, X_orig, X_perturbed; norm_numerator, norm_denominator, kwargs...)

Computes the ratio `norm_numerator(A_orig - A_perturbed) / norm_denominator(A_orig)`.
"""
function sensitivity_ratio(
        A_orig::AbstractMatrix{T}, A_perturbed::AbstractMatrix{T},
        X_orig::AbstractMatrix{T}, X_perturbed::AbstractMatrix{T};
        norm_numerator, norm_denominator,
        kwargs...
    ) where {T}

    numerator   = norm_numerator(A_orig .- A_perturbed)
    denominator = norm_denominator(A_orig)
    
    ratio = ifelse.(denominator .== 0, T(NaN), numerator ./ denominator)

    return vec(ratio)
end
