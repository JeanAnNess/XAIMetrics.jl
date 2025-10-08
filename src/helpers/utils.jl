"""
    expand_indices(arr::AbstractArray, indices, indexed_axes) -> Tuple

Creates a full tuple of indices for slicing an array `arr`, by placing the given `indices`
at the dimensions specified by `indexed_axes`. All other dimensions are indexed with a `Colon` (`:`),
representing a full slice.
"""
function expand_indices(
    arr::AbstractArray,
    indices::Tuple,
    indexed_axes::AbstractVector{<:Integer}
)
    nd = ndims(arr)
    sorted_axes = sort(collect(indexed_axes))

    if !allunique(sorted_axes)
        throw(ArgumentError("indexed_axes must be unique."))
    end
    if !all(1 .<= ax <= nd for ax in sorted_axes)
        throw(ArgumentError("All indexed_axes must be valid dimensions of the array."))
    end
    if length(indices) != length(sorted_axes)
        throw(DimensionMismatch("The number of indices must match the number of indexed_axes."))
    end

    full_indices = Vector{Any}(undef, nd)
    
    fill!(full_indices, :)

    # Place indices at specified axes
    for (ax, idx) in zip(sorted_axes, indices)
        full_indices[ax] = idx
    end

    return tuple(full_indices...)
end


function expand_indices(
    arr::AbstractArray,
    linear_indices::AbstractVector{<:Integer},
    indexed_axes::AbstractVector{<:Integer}
)
    sorted_axes = sort(collect(indexed_axes))
    subspace_dims = tuple((size(arr, d) for d in sorted_axes)...)
    cartesian_indices = CartesianIndices(subspace_dims)[linear_indices]

    unraveled_coords = ntuple(length(sorted_axes)) do i
        getindex.(cartesian_indices, i)
    end

    return expand_indices(arr, unraveled_coords, sorted_axes)
end

function expand_indices(
    arr::AbstractArray,
    linear_index::Integer,
    indexed_axes::AbstractVector{<:Integer}
)
    return expand_indices(arr, [linear_index], indexed_axes)
end