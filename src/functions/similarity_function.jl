using LinearAlgebra

"""
    distance_euclidean(a::AbstractArray, b::AbstractArray)

Calculate Euclidean distance between two arrays (e.g., images or explanations).
"""
function distance_euclidean(a::AbstractArray, b::AbstractArray)
    return norm(a-b)
end

"""
    lipschitz_constant(a, b, c, d; norm_numerator, norm_denominator)

Calculate the batched Lipschitz constant.
This function expects flattened matrices where each row corresponds to a sample.
"""
function lipschitz_constant(
    a::AbstractMatrix,
    b::AbstractMatrix,
    c::AbstractMatrix,
    d::AbstractMatrix;
    kwargs...
)
    eps = 1e-10

    default_dist = (x, y) -> norm(x - y)
    d1 = get(kwargs, :norm_numerator, default_dist)
    d2 = get(kwargs, :norm_denominator, default_dist)

    num_samples = size(a, 1)
    scores = zeros(eltype(a), num_samples)
    for i in axes(a, 1)
        a_row = @view a[i, :]
        b_row = @view b[i, :]
        c_row = @view c[i, :]
        d_row = @view d[i, :]

        numerator = d1(a_row, b_row)
        denominator = d2(c_row, d_row)
        scores[i] = numerator / (denominator + eps)
    end

    return scores
end

