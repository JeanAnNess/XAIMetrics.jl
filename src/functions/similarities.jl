"""
    distance_euclidean(a::AbstractArray, b::AbstractArray)

Calculate Euclidean distance between two arrays (e.g., images or explanations).
"""
function distance_euclidean(a::AbstractArray, b::AbstractArray)
    return norm(a - b)
end

"""
    distance_manhattan(a::AbstractArray, b::AbstractArray)

Calculate Manhattan (L1) distance between two arrays.
"""
function distance_manhattan(a::AbstractArray, b::AbstractArray)
    return sum(abs.(a .- b))
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

    num_samples = size(a, 2)
    scores = zeros(eltype(a), num_samples)

    for i in 1:num_samples
        a_col = @view a[:, i]
        b_col = @view b[:, i]
        c_col = @view c[:, i]
        d_col = @view d[:, i]

        numerator = norm_numerator(a_col, b_col)
        denominator = norm_denominator(c_col, d_col)
        scores[i] = numerator / (denominator + epsilon)
    end

    return scores
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
    sensitivities = A_orig - A_perturbed

    numerator = norm_numerator(sensitivities)
    denominator = norm_denominator(A_orig)

    ratio = numerator ./ denominator
    ratio[denominator .== 0] .= T(NaN)

    return vec(ratio)
end
