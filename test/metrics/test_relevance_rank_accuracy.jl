using Test
using XAIMetrics

using Random
using Statistics: mean

using Lux
using ExplainableAI

rng = Xoshiro(42)
model = Lux.Dense(784 => 10)
ps, st = Lux.setup(rng, model)
forward = x -> first(model(reshape(x, 784, :), ps, st))
analyzer = InputTimesGradient(forward, AutoForwardDiff())

@testset "RelevanceRankAccuracy" begin
    rng = Xoshiro(123)
    batch_size = 4
    x_batch = rand(rng, Float32, 28, 28, 1, batch_size)
    y_batch = [2, 5, 1, 9]

    @testset "Score Direction" begin
        @test scoredirection(RelevanceRankAccuracy()) === higherisbetter
    end

    @testset "Basic Execution" begin
        s_batch = rand(rng, Bool, 28, 28, 1, batch_size)
        metric = RelevanceRankAccuracy()
        scores = evaluate(metric, analyzer, x_batch; y=y_batch, s=s_batch)

        @test scores isa Vector{Float32}
        @test length(scores) == batch_size
        @test all(0.0f0 .<= scores .<= 1.0f0)
    end

    @testset "Normalization Config" begin
        s_batch = rand(rng, Bool, 28, 28, 1, batch_size)

        metric_norm = RelevanceRankAccuracy(normalize_config=NormalizationConfig(true))
        scores_norm = evaluate(metric_norm, analyzer, x_batch; y=y_batch, s=s_batch)

        metric_no_norm = RelevanceRankAccuracy(normalize_config=NormalizationConfig(false))
        scores_no_norm = evaluate(metric_no_norm, analyzer, x_batch; y=y_batch, s=s_batch)

        @test scores_norm isa Vector{Float32}
        @test scores_no_norm isa Vector{Float32}
        @test length(scores_norm) == batch_size
        @test length(scores_no_norm) == batch_size
    end

    @testset "Deterministic known values" begin
        metric = RelevanceRankAccuracy(normalize_config=NormalizationConfig(false))

        # Known ranking: explanations [0.1, 0.5, 0.3, 0.7], mask at [1, 4]
        # k=2, top-2 attribution indices: 4 (0.7), 2 (0.5)
        # mask[4]=true, mask[2]=false -> hits=1, score=1/2=0.5
        a = Float32[0.1f0, 0.5f0, 0.3f0, 0.7f0]
        s = Bool[true, false, false, true]
        s1 = XAIMetrics.relevance_rank_accuracy(metric, reshape(a, 4, 1), reshape(s, 4, 1))
        @test s1[1] == 0.5f0

        # Mask covers everything -> all top-k are in mask -> score = 1.0
        a2 = Float32[0.1f0, 0.5f0, 0.3f0]
        s2 = Bool[true, true, true]
        s2_out = XAIMetrics.relevance_rank_accuracy(metric, reshape(a2, 3, 1), reshape(s2, 3, 1))
        @test s2_out[1] == 1.0f0

        # Top attribution misses the mask entirely -> score = 0.0
        a3 = Float32[10.0f0, 0.0f0, 0.0f0, 0.0f0]
        s3 = Bool[false, true, false, false]
        s3_out = XAIMetrics.relevance_rank_accuracy(metric, reshape(a3, 4, 1), reshape(s3, 4, 1))
        @test s3_out[1] == 0.0f0

        # Top attribution hits mask -> all k=1 pixels in top-1 -> score = 1.0
        a4 = Float32[5.0f0, 1.0f0]
        s4 = Bool[true, false]
        s4_out = XAIMetrics.relevance_rank_accuracy(metric, reshape(a4, 2, 1), reshape(s4, 2, 1))
        @test s4_out[1] == 1.0f0
    end

    @testset "All-mask through full pipeline" begin
        # When the segmentation mask covers every pixel, 
        # all top-k attributions are necessarily in the mask -> score = 1.0
        s_all = trues(28, 28, 1, batch_size)
        scores = evaluate(RelevanceRankAccuracy(), analyzer, x_batch; y=y_batch, s=s_all)
        @test all(scores .== 1.0f0)
    end

    @testset "Empty Segmentation Mask" begin
        s_batch = falses(28, 28, 1, batch_size)
        scores = evaluate(RelevanceRankAccuracy(), analyzer, x_batch; y=y_batch, s=s_batch)
        @test scores isa Vector{Float32}
        @test all(isnan, scores)
        @test length(scores) == batch_size
    end

    @testset "Error When s is Missing" begin
        metric = RelevanceRankAccuracy()
        @test_throws ErrorException evaluate(metric, analyzer, x_batch; y=y_batch)
    end

    @testset "Error When s Size Mismatches" begin
        s_wrong = rand(rng, Bool, 14, 14, 1, batch_size)
        metric = RelevanceRankAccuracy()
        @test_throws AssertionError evaluate(metric, analyzer, x_batch; y=y_batch, s=s_wrong)
    end
end
