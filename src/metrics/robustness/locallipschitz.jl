"""
    LocalLipschitzEstimate(; nr_samples=200, similarity_func, norm_numerator, norm_denominator, perturb_std=0.1, return_nan_when_prediction_changes=false)
    <: AbstractRobustnessMetric

Robustness metric that tests the consistency in the explanation for neighboring examples
by estimating the local Lipschitz constant.
"""
@kwdef struct LocalLipschitzEstimate{FS, FN, FD} <: AbstractRobustnessMetric
    nr_samples::Int = 200
    similarity_func::FS = lipschitz_constant
    norm_numerator::FN = DEFAULT_NORM_FUNC
    norm_denominator::FD = DEFAULT_NORM_FUNC
    return_nan_when_prediction_changes::Bool = false
    perturb_config::PerturbationConfig = PerturbationConfig(gaussian_perturbation!, (; std = 0.2))
    normalize_config::NormalizationConfig = NormalizationConfig()
end

scoredirection(::LocalLipschitzEstimate) = lowerisbetter

"""
    evaluate(...)

Internal function to compute the Local Lipschitz Estimate for a batch of data.
"""
function evaluate(
        metric::LocalLipschitzEstimate,
        method::AbstractXAIMethod,
        x::AbstractArray{T, N},
        y::AbstractVector{<:Integer}, # true labels
        y_pred::AbstractMatrix{<:Real}, # predicted logits
        a::AbstractArray{T, N};
        s::Union{Nothing, AbstractArray{Bool, N}} = nothing
    ) where {T, N}

    return local_lipschitz_estimate(
        metric,
        method,
        x,
        y,
        y_pred,
        a
    )
end

function local_lipschitz_estimate(
        metric::LocalLipschitzEstimate,
        method::AbstractXAIMethod,
        x::AbstractArray{T, N},
        y::AbstractVector{<:Integer},
        y_pred::AbstractMatrix{<:Real},
        a::AbstractArray{T, N}
    ) where {T, N}
    # model = method.model
    _size = size(x)
    batch_size = _size[end]
    num_features = prod(_size[1:(end - 1)])

    # Compute initial explanations
    a_processed = normalize_explanations(a, metric.normalize_config)

    # Flatten for similarity computation (features × batch)
    X_orig_flat = reshape(x, num_features, batch_size)
    A_orig_flat = reshape(a_processed, num_features, batch_size)

    similarities = Matrix{T}(undef, batch_size, metric.nr_samples)

    x_perturbed = similar(x)
    X_perturbed_flat = similar(X_orig_flat)

    for i in 1:metric.nr_samples
        perturb_input!(x_perturbed, x, metric.perturb_config)

        expl_perturbed = analyze(x_perturbed, method, IndexSelector(y))
        a_perturbed = expl_perturbed.val
        a_perturbed_processed = normalize_explanations(a_perturbed, metric.normalize_config)

        # Predictions for perturbed batch
        changed_idx = falses(batch_size)
        if metric.return_nan_when_prediction_changes
            y_pred_classes = predicted_classes(y_pred)
            y_pred_perturbed = predicted_classes(expl_perturbed.output)
            changed_idx .= y_pred_classes .!= y_pred_perturbed
        end

        A_perturbed_flat = reshape(a_perturbed_processed, num_features, batch_size)
        X_perturbed_flat = reshape(x_perturbed, num_features, batch_size)

        sim_scores = metric.similarity_func(
            A_orig_flat, A_perturbed_flat,
            X_orig_flat, X_perturbed_flat;
            norm_numerator = metric.norm_numerator,
            norm_denominator = metric.norm_denominator
        )

        # Mask changed predictions with NaN
        sim_scores[changed_idx] .= T(NaN)
        similarities[:, i] = sim_scores
    end

    # Replace remaining NaNs with -Inf if not returning NaNs
    if !metric.return_nan_when_prediction_changes
        similarities[isnan.(similarities)] .= T(-Inf)
    end

    scores = dropdims(maximum(similarities, dims = 2), dims = 2)
    return scores
end
