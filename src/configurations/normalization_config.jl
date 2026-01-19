# Functions

"""
    normalize_by_max_abs(a::AbstractArray)
    normalize_by_max_abs(a::AbstractArray, dims)

Normalize an array by dividing by its maximum absolute value, scaling values to [-1, 1]. 
Optionally, normalization can be performed along specified dimensions `dims`.

# Examples
```julia-repl
julia> A = [1.0 -2.0; 3.0 -4.0]
2×2 Matrix{Float64}:
 1.0  -2.0
 3.0  -4.0

julia> normalize_by_max_abs(A)
2×2 Matrix{Float64}:
 0.25  -0.5
 0.75  -1.0

julia> normalize_by_max_abs(A, 1)
2×2 Matrix{Float64}:
 0.333333  -0.5
 1.0       -1.0
```
"""
function normalize_by_max_abs(a::AbstractArray, dims)
    all(iszero, a) && return a
    return a ./ maximum(abs, a; dims)
end

function normalize_by_max_abs(a::AbstractArray)
    all(iszero, a) && return a
    return a ./ maximum(abs, a)
end

# Struct
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
