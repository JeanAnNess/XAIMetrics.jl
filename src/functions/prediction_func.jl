"""
    predicted_classes(output::AbstractMatrix) -> Vector{Int}

Extract predicted class indices from a matrix of model outputs (logits or probabilities).

# Arguments
- `output::AbstractMatrix`: A matrix where each column contains the model outputs for one sample.

# Returns
- `Vector{Int}`: A vector of predicted class indices (one per sample).

# Examples
```julia
logits = [0.1 0.8; 0.9 0.2]
predicted_classes(logits)  # [2, 1]
```
"""
function predicted_classes(output::AbstractMatrix)
    return [argmax(col) for col in eachcol(output)]
end
