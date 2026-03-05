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
function normalize_by_max_abs(a::AbstractArray{T}, dims) where {T}
    all(iszero, a) && return a
    return a ./ maximum(abs, a; dims)
end

function normalize_by_max_abs(a::AbstractArray)
    all(iszero, a) && return a
    return a ./ maximum(abs, a)
end

"""
	stable_division(numerator, denominator)

Perform division with an epsilon-stabilized denominator to reduce division-by-zero
instabilities for scalar and array inputs.
"""
function stable_division(numerator::AbstractArray{T}, denominator::AbstractArray{T}) where {T}
    S = float(T)
    epsilon = eps(S)
    return S.(numerator) ./ (S.(denominator) .+ epsilon)
end

function stable_division(numerator::Number, denominator::Number)
    S = float(promote_type(typeof(numerator), typeof(denominator)))
    epsilon = eps(S)
    return S(numerator) / (S(denominator) + epsilon)
end

function stable_division(numerator::AbstractArray, denominator::Number)
    S = float(promote_type(eltype(numerator), typeof(denominator)))
    epsilon = eps(S)
    return S.(numerator) ./ (S(denominator) .+ epsilon)
end

function stable_division(numerator::Number, denominator::AbstractArray)
    S = float(promote_type(typeof(numerator), eltype(denominator)))
    epsilon = eps(S)
    return S(numerator) ./ (S.(denominator) .+ epsilon)
end
