using XAIMetrics
using Aqua
using JET
using Test

@testset "Aqua.jl" begin
    Aqua.test_all(XAIMetrics)
end

@testset "JET.jl" begin
    JET.test_package(XAIMetrics; target_modules=(XAIMetrics,))
end