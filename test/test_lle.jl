using Random
using LinearAlgebra
using Flux
using ExplainableAI

# model and analyzer
flux_model = Chain(
    Conv((3, 3), 1 => 4, relu ; pad=(1, 1)), 
    Flux.flatten,
    Dense(4 * 28 * 28, 10)
)

analyzer = InputTimesGradient(flux_model)

@testset "Local Lipschitz Estimate Integration" begin
    Random.seed!(123)
    batch_size = 4
    x_batch = rand(Float32, 28, 28, 1, batch_size)
    y_batch = [2, 5, 1, 9]

    perturb_cfg_low_std = PerturbationConfig(perturb_std=0.1)
    perturb_cfg_high_std = PerturbationConfig(perturb_std=0.8)
    
    @testset "Basic Execution with Analyzer" begin
        metric_low_std = LocalLipschitzEstimate(
            nr_samples=50, 
            perturb_config=perturb_cfg_low_std
        )
        
        metric_high_std = LocalLipschitzEstimate(
            nr_samples=50, 
            perturb_config=perturb_cfg_high_std
        )

        scores_low = evaluate(metric_low_std, analyzer, x_batch, y=y_batch)
        scores_high = evaluate(metric_high_std, analyzer, x_batch, y=y_batch)

        println("Scores with low std: ", scores_low)
        println("Scores with high std: ", scores_high)
        @test scores_low isa Vector{Float32}
        @test scores_high isa Vector{Float32}
    end

    @testset "Prediction Change Check with Analyzer" begin
        metric_low_std_with_check = LocalLipschitzEstimate(
            nr_samples=50,
            perturb_config=perturb_cfg_low_std,
            return_nan_when_prediction_changes=true
        )

        metric_high_std_with_check = LocalLipschitzEstimate(
            nr_samples=50,
            perturb_config=perturb_cfg_high_std,
            return_nan_when_prediction_changes=true
        )
        
        scores_low = evaluate(
            metric_low_std_with_check, analyzer, x_batch, y=y_batch
        )

        scores_high = evaluate(
            metric_high_std_with_check, analyzer, x_batch, y=y_batch
        )
        println("Scores with low std (with check): ", scores_low)
        println("Scores with high std (with check): ", scores_high)
        
        @test any(isnan, scores_low)
        @test any(isnan, scores_high)
    end
end