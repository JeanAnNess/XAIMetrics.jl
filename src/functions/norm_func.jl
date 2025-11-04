function frobenius_norm(a::AbstractArray{T, N}) where {T, N}
    @assert N == 1 || N == 2 "frobenius_norm did not receive a 1D or 2D array."
    return sqrt(sum(abs2, a))
end

function columnwise_l2_norm(A::AbstractMatrix{T}) where T
    batch_size = size(A, 2)
    norms = Vector{T}(undef, batch_size)

    for i in 1:batch_size
        col_view = @view A[:, i]
        norms[i] = frobenius_norm(col_view)
    end
    return norms'
end