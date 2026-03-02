using XAIMetrics
using ExplainableAI
using Flux
using Random

model = Chain(
    Flux.flatten,
    Dense(28 * 28, 64, relu),
    Dense(64, 10),
)

method = InputTimesGradient(model)

batch_size = 8
x = rand(Float32, 28, 28, 1, batch_size)
y = rand(1:10, batch_size)

metric = AvgSensitivity(
    nr_samples = 30,
    perturb_config = PerturbationConfig(
        uniform_noise!;
        params = (; lower = 0.05, upper = 0.1),
    ),
)

scores = evaluate(metric, method, x; y = y)

lle_metric = LocalLipschitzEstimate(
    nr_samples = 50,
    perturb_config = PerturbationConfig(
        gaussian_perturbation!;
        params = (; std = 0.2),
    ),
    return_nan_when_prediction_changes = false,
)

lle_scores = evaluate(lle_metric, method, x; y = y)

cfg_uniform = PerturbationConfig(
    uniform_noise!;
    params = (; lower = 0.03, upper = 0.08))

norm_cfg = NormalizationConfig(
    normalize = true,
    normalize_func = normalize_by_max_abs,
)

metric_norm = AvgSensitivity(
    nr_samples = 30,
    perturb_config = cfg_uniform,
    normalize_config = norm_cfg,
)

Random.seed!(7)

x_demo = Float32[0.2, -0.4, 0.8]
x_demo_pert = similar(x_demo)

demo_pert_cfg = PerturbationConfig(uniform_noise!; params = (; lower = 0.03, upper = 0.08))
perturb_input!(x_demo_pert, x_demo, demo_pert_cfg)

println("x_demo          = ", x_demo)
println("x_demo_pert     = ", x_demo_pert)
println("perturbation Δ  = ", x_demo_pert .- x_demo)

a_demo = Float32[-2.0, 0.0, 1.0, 4.0]
a_norm = normalize_by_max_abs(a_demo)

println("a_demo                  = ", a_demo)
println("normalize_by_max_abs    = ", a_norm)
println("maximum(abs, a_demo)    = ", maximum(abs, a_demo))
println("maximum(abs, a_norm)    = ", maximum(abs, a_norm))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
