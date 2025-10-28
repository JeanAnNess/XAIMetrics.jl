using Test
using XAIMetrics

@testset "XAIMetrics.jl" begin
    include("test_lle.jl")
    include("test_normalize.jl")
end
