using Test
using XAIMetrics

@testset "XAIMetrics.jl" begin
	include("test_lle.jl")
    include("test_normalize.jl")
    include("test_similarity_func.jl")
    include("test_avg_sens.jl")
	include("test_qa.jl")
end
