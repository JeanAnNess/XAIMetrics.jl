# TODO: add more perturb functions, add normalization functions
frobenius_norm(a::AbstractVecOrMat) = sqrt(sum(abs2, a)) # p = 2 norm, >TODO: nachschauen, ggf funktion raus

columnwise_l2_norm(A::AbstractMatrix) = sqrt.(sum(abs2, A; dims=1)) 

