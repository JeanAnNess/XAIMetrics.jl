import XAIBase: analyze, Explanation, MaxActivationSelector
import ExplainableAI: Gradient, GradCAM
const XAI_ANALYZER_MAP = Dict(
    :Gradient => Gradient,         # Saliency Maps
    :GradCAM => GradCAM,          # Grad-CAM
)

const TEST_MODEL = Flux.Chain(Flux.flatten, Flux.Dense(4 * 4 * 3 => 5, Flux.relu), Flux.Dense(5 => 3))
const DUMMY_INPUT_SHAPE = (4, 4, 3, 8)
const DUMMY_BATCH_SIZE = DUMMY_INPUT_SHAPE[end]

function generate_test_data()
    x = rand(Float32, DUMMY_INPUT_SHAPE...)
    y = rand(1:3, DUMMY_BATCH_SIZE)
    return x, y
end

function get_real_explanation(model, x_batch, y_batch; analyzer_name::Symbol = :Gradient, kwargs...)
    AnalyzerType = get(XAI_ANALYZER_MAP, analyzer_name) do
        @warn "Analyzer '$analyzer_name' not found in map. Defaulting to Gradient()."
        return Gradient
    end

    analyzer = AnalyzerType(model)

    target_class = y_batch[1]
    output_selector = MaxActivationSelector()
    x_instance = selectdim(x_batch, ndims(x_batch), 1)
    x_instance = reshape(x_instance, :, 1)

    return analyze(x_instance, analyzer, output_selector)
end
