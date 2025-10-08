"""
    normalize_by_max_abs(batch::AbstractArray)

Normalizes each sample in a batch by dividing by its maximum absolute value.
This scales each sample to the range [-1, 1].
"""
function normalize_by_max_abs(
    a::AbstractArray{T, N},
    normalise_axes::Union{Nothing, AbstractVector{Int}}=nothing
) where {T, N}
    
    if all(a .== 0.0)
        return a
    end

    if isnothing(normalise_axes)
        normalise_axes = collect(1:N)
    end

    max_abs = maximum(abs, a; dims=normalise_axes)
    
    return a ./ max_abs
end