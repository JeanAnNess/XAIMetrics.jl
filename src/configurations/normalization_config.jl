struct NormalizationConfig
    normalize::Bool
    normalize_func::Function

    NormalizationConfig(; normalize::Bool = true, normalize_func::Function = normalize_by_max_abs) = new(normalize, normalize_func)
end

function normalize_explanations(
        a_batch::AbstractArray{T, N},
        config::NormalizationConfig
    ) where {T, N}
    if config.normalize
        return config.normalize_func(a_batch)
    else
        return a_batch
    end
end

# Functions

"""
    normalize_by_max_abs(batch::AbstractArray)

Normalizes each sample in a batch by dividing by its maximum absolute value.
This scales each sample to the range [-1, 1].
"""
function normalize_by_max_abs(a::AbstractArray, dims)
    all(iszero, a) && return a
    return a / maximum(abs, a; dims)
end

function normalize_by_max_abs(a::AbstractArray)
    all(iszero, a) && return a
    return a / maximum(abs, a)
end