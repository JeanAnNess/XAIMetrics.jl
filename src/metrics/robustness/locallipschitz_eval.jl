using Random
using Statistics
using XAIBase: AbstractXAIMethod, analyze

"""
    _evaluate(...)

Internal function to compute the Local Lipschitz Estimate for a batch of data.
"""
function _evaluate(
    metric::LocalLipschitzEstimate,
    model, 
    x_batch::AbstractArray{T, N}, 
    y_batch::AbstractVector{<:Integer},
    a_batch::AbstractArray{T, N};
    explain_func::Function,
    return_internals::Bool=false
) where {T<:AbstractFloat, N}

    if isnothing(explain_func)
        error("LocalLipschitzEstimate requires an 'explain_func' to re-compute explanations for perturbed inputs.")
    end

    # (W, H, C, B)
    batch_size = size(x_batch)[end]
    num_features = div(length(x_batch), batch_size)

    a_batch_processed = similar(a_batch)
    # TODO: Auslagern von normalise
    if metric.normalise
        for i in axes(a_batch, 4)
            sample = @view a_batch[:,:,:,i]
            max_abs = maximum(abs, sample)
            a_batch_processed[:,:,:,i] = sample ./ (max_abs + 1e-10)
        end
    else
        a_batch_processed = a_batch
    end

    # flatten to (batch_size, features), (B, W*H*C)
    X_orig_flat = reshape(x_batch, num_features, batch_size)'
    A_orig_flat = reshape(a_batch_processed, num_features, batch_size)'

    similarities = Matrix{T}(undef, batch_size, metric.nr_samples)

    # original predictions
    y_pred_orig_idx = nothing
    if metric.return_nan_when_prediction_changes
        # logits (num_classes, batch_size)
        y_pred_logits = model(x_batch)
        y_pred_orig_idx = vec(mapslices(argmax, y_pred_logits, dims=1)) # index of max logit per sample
    end

    for i in 1:metric.nr_samples
        # perturb input
        noise = T(metric.perturb_std) .* randn(T, size(x_batch))
        x_perturbed = x_batch .+ noise

        # check if prediction changed
        changed_idx = falses(batch_size)
        if metric.return_nan_when_prediction_changes # Todo: auslagern
            y_pred_pert_logits = model(x_perturbed)
            y_pred_pert_idx = vec(mapslices(argmax, y_pred_pert_logits, dims=1))
            changed_idx = (y_pred_orig_idx .!= y_pred_pert_idx)
			# print("Changed indices in sample $i: ", (changed_idx), "\n")
        end

        # print("Sample $i: Perturbed inputs for $(sum(changed_idx)) out of $batch_size samples changed the prediction.\n")

        # explanation for perturbed input
        # Assumes explain_func returns a similar structure/array
        a_perturbed = explain_func(model, x_perturbed, y_batch)

        a_perturbed_processed = similar(a_perturbed)

        # TODO: Auslagern von normalise
        if metric.normalise
            for j in axes(a_perturbed, 4)
                sample = @view a_perturbed[:,:,:,j]
                max_abs = maximum(abs, sample)
                a_perturbed_processed[:,:,:,j] = sample ./ (max_abs + 1e-10)
            end
        else
            a_perturbed_processed = a_perturbed
        end

        A_perturbed_flat = reshape(a_perturbed_processed, num_features, batch_size)'

        # similarity
        X_perturbed_flat = reshape(x_perturbed, num_features, batch_size)'

        sim_scores = metric.similarity_func(
            A_orig_flat, A_perturbed_flat,
            X_orig_flat, X_perturbed_flat;
            norm_numerator = metric.norm_numerator,
            norm_denominator = metric.norm_denominator
        )
        
        # print("Sample $i: Similarity scores computed: $sim_scores\n")

        # masking changed predictions with NaN
        sim_scores[changed_idx] .= T(NaN)
        similarities[:, i] = sim_scores
        # print("Sample $i: Similarity scores after masking: $(similarities[:, i])\n")
    end

    # print("All similarities matrix:\n", similarities, "\n")

    local scores::Vector{T}
    if !metric.return_nan_when_prediction_changes
        similarities[isnan.(similarities)] .= T(-Inf)
    end
    
    scores = dropdims(maximum(similarities, dims=2), dims=2)
    return return_internals ? (scores, similarities) : scores

end

function _evaluate(
    metric::LocalLipschitzEstimate,
    analyzer::AbstractXAIMethod,
    x_batch::AbstractArray{T, N},
    y_batch::AbstractVector{<:Integer},
    a_batch::AbstractArray{T, N};
    return_internals::Bool=false
) where {T<:AbstractFloat, N}
    
    model = analyzer.model
    
    # wrapper for explain_func that uses the analyzer
    function explain_func_wrapper(m, current_x_batch, current_y_batch)
        batch_size = size(current_x_batch)[end]
        explanations = similar(current_x_batch)

        for i in 1:batch_size
            input_batch = current_x_batch[:, :, :, i:i]
            target_index = current_y_batch[i]
            
            # Call analyze
            explanation_struct = analyze(input_batch, analyzer, IndexSelector(target_index)) # Todo: Maybe have the selector as a parameter

            # output is (W, H, C, 1), want (W, H, C)
            explanations[:, :, :, i] = explanation_struct.val[:, :, :, 1]
        end
        return explanations
    end
    
    return _evaluate(
        metric,
        model,
        x_batch,
        y_batch,
        a_batch;
        explain_func=explain_func_wrapper,
        return_internals=return_internals
    )
end

function _evaluate(
    metric::LocalLipschitzEstimate,
    analyzer::AbstractXAIMethod,
    x_batch::AbstractArray{T, N},
    y_batch::AbstractVector{<:Integer};
    return_internals::Bool=false
) where {T<:AbstractFloat, N}

    # 1. Generate the initial explanation batch automatically.
    a_batch = similar(x_batch)
    for i in 1:size(x_batch)[end]
        input_4D = x_batch[:, :, :, i:i]
        explanation = analyze(input_4D, analyzer, IndexSelector(y_batch[i]))
        # print(explanation.val.size, "\n")
        a_batch[:, :, :, i] = explanation.val[:, :, :, 1]
    end

    return _evaluate(
        metric, analyzer, x_batch, y_batch, a_batch;
        return_internals=return_internals
    )
end