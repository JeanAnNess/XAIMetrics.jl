using Random

"""
    gaussian_noise(
        arr::AbstractArray,
        indices,
        indexed_axes::AbstractVector{<:Integer};
        mean::Real = 0.0,
        std::Real = 0.01
    ) -> AbstractArray

Adds Gaussian noise to a specified subset of an array.
"""
function gaussian_noise(
    arr::AbstractArray,
    indices,
    indexed_axes::AbstractVector{<:Integer};
    mean::Real = 0.0,
    std::Real = 0.01,
    kwargs...
)
    # full index tuple
    expanded_idx = expand_indices(arr, indices, indexed_axes)

    # copy
    arr_perturbed = copy(arr)

    # view into the specific slice 
    slice_to_perturb = view(arr_perturbed, expanded_idx...)

    noise = randn(size(slice_to_perturb)) .* std .+ mean
    slice_to_perturb .+= noise

    return arr_perturbed
end