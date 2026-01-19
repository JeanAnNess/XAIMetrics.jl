function frobenius_norm(a::AbstractArray{T, N}) where {T, N}
    @assert N == 1 || N == 2 "frobenius_norm did not receive a 1D or 2D array."
    return sqrt(sum(abs2, a))
end

function columnwise_l2_norm(A::AbstractMatrix{T}) where {T}
    norms = [sqrt(sum(abs2, col)) for col in eachcol(A)]
    return norms'
end
