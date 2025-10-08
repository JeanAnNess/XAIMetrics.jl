using Base: @kwdef

"""
    QuantusPixelFlipping(; perturb_baseline="black", abs=true) 
    <: AbstractFaithfulnessMetric

Faithfulness metric that measures the drop in model confidence when features 
are progressively removed/masked in order of their attribution score.
"""
@kwdef struct PixelFlipping{T} <: AbstractFaithfulnessMetric
    perturb_baseline::T = "black"
    abs::Bool = true
    steps::Int = 20
end

const AbstractWHCN{T} = AbstractArray{T, 4} where T 
const AbstractBatch{T} = AbstractArray{T, N} where {T, N}