using Test
using XAIMetrics

@testset "XAIMetrics.jl" begin
	include("test_qa.jl")
    include("test_similarity_func.jl")
    include("test_normalize.jl")
    # Metrics
	include("test_lle.jl")
    include("test_avg_sens.jl")
    include("robustness_parity.jl")
end
