frobenius_norm(a::AbstractVecOrMat) = sqrt(sum(abs2, a)) # p = 2 norm, >TODO: nachschauen, ggf funktion raus

function columnwise_l2_norm(A::AbstractMatrix)
    norms = [sqrt(sum(abs2, col)) for col in eachcol(A)]
    return norms' # Return nicht transponieren, sondern dann den call transponieren
end

# norm.(eachcol(A), 2)
