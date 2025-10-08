using Test
using ExplanationMetrics

@testset "ExplanationMetrics.jl" begin
    include("test_lle.jl")
    include("test_normalize.jl")
end
