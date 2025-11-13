using LinearAlgebra

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

"""
function lipschitz_constant(
    a::AbstractMatrix,
    b::AbstractMatrix,
    c::AbstractMatrix,
    d::AbstractMatrix;
    norm_numerator = distance_manhattan,
    norm_denominator = distance_euclidean
)
    eps = 1e-10
    default_dist = (x, y) -> norm(x - y)
    d1 = norm_numerator
    d2 = norm_denominator

    num_samples = size(a, 2) 
    scores = zeros(eltype(a), num_samples)

    for i in 1:num_samples
        a_col = @view a[:, i]
        b_col = @view b[:, i]
        c_col = @view c[:, i]
        d_col = @view d[:, i]

        numerator = d1(a_col, b_col)
        denominator = d2(c_col, d_col)
        scores[i] = numerator / (denominator + eps)
    end

    return scores
end

"""
    difference(a, b)

Calculate the difference between two images or explanations.
"""
function difference(
    a::AbstractMatrix,
    b::AbstractMatrix
)
    return a-b
end

