"""
    NormalizationConfig(; normalize=true, normalize_func=normalize_by_max_abs)

Configuration for explanation normalization in metric pipelines.

- `normalize`: enables/disables normalization
- `normalize_func`: callable applied to explanation batches when normalization is enabled
"""
struct NormalizationConfig{F<: Function}
    normalize::Bool
    normalize_func::F # siehe perturbations

    NormalizationConfig(; normalize::Bool = true, normalize_func::F = normalize_by_max_abs) where {F} = new{F}(normalize, normalize_func)
end

"""
    normalize_explanations(a_batch, config)

Apply normalization to an explanation batch according to `config`.
Returns the original batch unchanged when `config.normalize == false`.
"""
function normalize_explanations(
        a_batch::AbstractArray,
        config::NormalizationConfig
    )
    if config.normalize
        return config.normalize_func(a_batch)
    else
        return a_batch
    end
end
