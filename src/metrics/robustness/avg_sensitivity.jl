"""
    AvgSensitivity

Compute the average sensitivity of a model with respect to its input features.

This metric measures how much the model's output changes on average, when
each feature of the input is slightly perturbed.
"""
@kwdef struct AvgSensitivity{FS, FN, FD} <: AbstractRobustnessMetric
    nr_samples::Int = 200
    similarity_func::FS = difference
    norm_numerator::FN = DEFAULT_SENS_NORM_FUNC
    norm_denominator::FD = DEFAULT_SENS_NORM_FUNC
    return_nan_when_prediction_changes::Bool = false
    perturb_config::PerturbationConfig = PerturbationConfig(perturb_func = uniform_noise!)
    normalize_config::NormalizationConfig = NormalizationConfig()
end

scoredirection(::AvgSensitivity) = lowerisbetter

"""
    evaluate(...)

Internal function to compute the AvgSensitivity for a batch of data.
"""
function evaluate(
        metric::AvgSensitivity,
        method::AbstractXAIMethod,
        x::AbstractArray{T, N},
        y::AbstractVector{<:Integer}, # true labels
        y_pred::AbstractMatrix{<:Real}, # predicted logits
        a::AbstractArray{T, N};
        s::Union{Nothing, AbstractArray{Bool, N}} = nothing
    ) where {T, N}

    return avg_sensitivity_estimate(
        metric,
        method,
        x,
        y,
        y_pred,
        a
    )
end

"""
    avg_sensitivity_estimate(...)

Computes the average sensitivity score.
"""
function avg_sensitivity_estimate(
        metric::AvgSensitivity,
        method::AbstractXAIMethod,
        x::AbstractArray{T, N},
        y::AbstractVector{<:Integer},
        y_pred::AbstractMatrix{<:Real},
        a::AbstractArray{T, N}
    ) where {T, N}

    _size = size(x)
    batch_size = _size[end]
    num_features = prod(_size[1:(end - 1)])

    # Compute initial explanations
    a_processed = normalize_explanations(a, metric.normalize_config)

    # Flatten for similarity computation (features x batch)
    A_orig_flat = reshape(a_processed, num_features, batch_size)

    similarities = Matrix{T}(undef, batch_size, metric.nr_samples)

    x_perturbed = similar(x)

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

        sensitivities = metric.similarity_func(A_orig_flat, A_perturbed_flat)
        numerator = metric.norm_numerator(sensitivities)
        denominator = metric.norm_denominator(A_orig_flat)

        sim_scores = numerator ./ denominator
        sim_scores[denominator .== 0] .= T(NaN) #handle zero div

        # Mask changed predictions with NaN
        sim_scores[changed_idx] .= T(NaN)
        similarities[:, i] = sim_scores
    end

    if metric.return_nan_when_prediction_changes
        scores = dropdims(mean(similarities, dims = 2), dims = 2)
    else
        scores = [mean(filter(!isnan, row)) for row in eachrow(similarities)]
    end
    return scores
end
