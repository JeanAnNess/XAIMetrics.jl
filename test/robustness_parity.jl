using Test
using XAIMetrics

using Random
using Flux
using ExplainableAI
using Statistics
using Metalhead
using JLD2

ENV["XAIMETRICS_DEBUG_PROGRESS"] = "false"
ENV["XAIMETRICS_DEBUG_EVERY"] = "1"

const ASSETS_DIR = joinpath(@__DIR__, "assets")
const SANITY_DATA_FILE = joinpath(ASSETS_DIR, "robustness.jld2")
println("Assets directory: $ASSETS_DIR")

@testset "Quantus Parity Checks" begin
    Random.seed!(123)
    data = load(SANITY_DATA_FILE)
    x_batch = data["x_batch"]       # (W, H, C, N) Float32
    y_batch = data["y_batch"]       # Vector{Int} 
    expected_preds = data["expected_preds"] # Vector{Int}
    quantus_as = data["quantus_avg_sensitivity"]
    quantus_lle = data["quantus_local_lipschitz"]
    
    # model
    model = ResNet(18; pretrain=true)
    testmode!(model)
    @info "Building InputTimesGradient analyzer" metric_samples=10
    analyzer = InputTimesGradient(model)

    # Prediction Parity
    @testset "Prediction Parity (ResNet18)" begin
        # Metalhead ResNet output: (Classes, Batch)
        logits = model(x_batch)
        preds = [argmax(col) for col in eachcol(logits)]
        # agreement to py
        agreement = mean(preds .== expected_preds)
        
        @info "Model Prediction Agreement: $(agreement * 100)%"
        @test agreement >= 0.9
    end

    @testset "Metric: Average Sensitivity" begin
        @info "Evaluating AvgSensitivity..."
        perturb_cfg = PerturbationConfig(uniform_noise!, (; lower = 0.2))
        metric = AvgSensitivity(
            nr_samples=10,
            perturb_config=perturb_cfg
        )
        scores = evaluate(metric, analyzer, x_batch, y=y_batch)
        @test all(isapprox.(scores, quantus_as, atol=0.3))  
        avg_tolerance = mean(abs.(scores .- quantus_as))
        @info "AvgSensitivity average absolute difference from Quantus: $avg_tolerance"
    end

    @testset "Metric: Local Lipschitz Estimate" begin
        @info "Evaluating LocalLipschitzEstimate..."
        perturb_cfg = PerturbationConfig(gaussian_perturbation!, (; std = 0.2))
        metric = LocalLipschitzEstimate(
            nr_samples=10,
            perturb_config=perturb_cfg
        )
        scores = evaluate(metric, analyzer, x_batch, y=y_batch)
        @test all(isapprox.(scores, quantus_lle, atol=0.3))
        avg_tolerance = mean(abs.(scores .- quantus_lle))
        @info "LocalLipschitzEstimate average absolute difference from Quantus: $avg_tolerance"
    end
end