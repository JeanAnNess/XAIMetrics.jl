struct PerturbationConfig
	perturb_std::Float64
	perturb_func::Function

    PerturbationConfig(; perturb_std::Float64=0.1, perturb_func::Function=gaussian_perturbation!) = new(perturb_std, perturb_func)
end

function gaussian_perturbation!(x_perturbed, x_batch, std)
    randn!(x_perturbed)	# fill with N(0,1)
    x_perturbed .= x_batch .+ std .* x_perturbed
end

function perturb_input!(
    x_perturbed::AbstractArray{T},
    x_batch::AbstractArray{T},
    config::PerturbationConfig
) where {T}
    config.perturb_func(x_perturbed, x_batch, config.perturb_std)
end