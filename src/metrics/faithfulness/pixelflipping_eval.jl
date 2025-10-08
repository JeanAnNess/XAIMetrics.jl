function _evaluate(
    metric::PixelFlipping, 
    model, 
    x_batch::AbstractWHCN{T}, 
    y_batch::AbstractVector{Int}, 
    a_batch::AbstractWHCN{T};
    kwargs...
    ) where {T}

    batch_size = size(x_batch)[end]
    num_features = div(length(x_batch), batch_size)
    
    if ndims(a_batch) != ndims(x_batch)
        @warn "Attribution shape mismatch. Assuming attributions are already pre-processed."
    end

    return rand(Float32, batch_size) 
end