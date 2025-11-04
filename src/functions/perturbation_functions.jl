function gaussian_perturbation!(x_perturbed, x_batch; std)
    randn!(x_perturbed)	# fill with N(0,1)
    x_perturbed .= x_batch .+ std .* x_perturbed
end

function uniform_noise!(x_perturbed, x_batch; lower, upper=nothing)
    if upper === nothing
        # symmetric range: [-lower, lower]
        x_perturbed .= x_batch .+ (-lower .+ (2 * lower) .* rand(size(x_batch)...))
    else
        x_perturbed .= x_batch .+ (lower .+ (upper - lower) .* rand(size(x_batch)...))
    end
end