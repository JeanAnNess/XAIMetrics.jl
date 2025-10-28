struct NormalizationConfig
	normalize::Bool
	normalize_func::Function

	NormalizationConfig(; normalize::Bool=true, normalize_func::Function=normalize_by_max_abs) = new(normalize, normalize_func)
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