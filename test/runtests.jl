using Test
using XAIMetrics

@testset "XAIMetrics.jl" begin
	include("test_qa.jl")
    include("test_similarity_func.jl")
    include("test_normalize.jl")
    # Metrics
	include("metrics/test_lle.jl")
    include("metrics/test_avg_sens.jl")
    # include("metrics/robustness_parity.jl")  need to rewrite using Lux
end
