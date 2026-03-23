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

@testset "AvgSensitivity Integration" begin
    rng = Xoshiro(123)
    batch_size = 4
    x_batch = rand(rng, Float32, 28, 28, 1, batch_size)
    y_batch = [2, 5, 1, 9]

    perturb_cfg_low_noise = PerturbationConfig(uniform_noise!, (lower = 0.1, upper = 0.2), rng = rng)
    perturb_cfg_high_noise = PerturbationConfig(uniform_noise!, (lower = 0.5, upper = 1.9), rng = rng)

    @testset "Basic Execution with Analyzer" begin
        metric_low_noise = AvgSensitivity(
            nr_samples = 50,
            perturb_config = perturb_cfg_low_noise
        )

        metric_high_noise = AvgSensitivity(
            nr_samples = 50,
            perturb_config = perturb_cfg_high_noise
        )

        scores_low = evaluate(metric_low_noise, analyzer, x_batch, y = y_batch)
        scores_high = evaluate(metric_high_noise, analyzer, x_batch, y = y_batch)

        println("Scores with low noise (AvgSens): ", scores_low)
        println("Scores with high noise (AvgSens): ", scores_high)

        @test scores_low isa Vector{Float32}
        @test scores_high isa Vector{Float32}
        @test !any(isnan, scores_low)   # default behavior should not return NaN
        @test !any(isnan, scores_high)
        @test mean(scores_high) > mean(scores_low) # Higher noise should increase sensitivity
    end

    @testset "Prediction Change Check with Analyzer" begin
        metric_low_noise_with_check = AvgSensitivity(
            nr_samples = 50,
            perturb_config = perturb_cfg_low_noise,
            return_nan_when_prediction_changes = true
        )

        metric_high_noise_with_check = AvgSensitivity(
            nr_samples = 50,
            perturb_config = perturb_cfg_high_noise,
            return_nan_when_prediction_changes = true
        )

        scores_low = evaluate(
            metric_low_noise_with_check, analyzer, x_batch, y = y_batch
        )

        scores_high = evaluate(
            metric_high_noise_with_check, analyzer, x_batch, y = y_batch
        )
        println("Scores with low noise (with check): ", scores_low)
        println("Scores with high noise (with check): ", scores_high)

        @test any(isnan, scores_high)
        @test count(isnan, scores_high) >= count(isnan, scores_low)
    end
end
