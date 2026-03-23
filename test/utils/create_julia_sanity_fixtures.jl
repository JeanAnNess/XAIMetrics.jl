using NPZ
using JLD2

const REPO_ROOT = dirname(dirname(@__DIR__))
const INTERMEDIATE_FILE = joinpath(REPO_ROOT, "test", "assets", "quantus_intermediate.npz")
const OUTPUT_FILE = joinpath(REPO_ROOT, "test", "assets", "sanity_batch.jld2")

@info "Loading intermediate NPZ..."
data = npzread(INTERMEDIATE_FILE)

# Extract data
# PyTorch: (N, C, H, W)
x_py = data["x_batch"] 
y_py = data["y_batch"]
logits_py = data["py_logits"]
preds_py = data["py_preds"]
scores_as = data["quantus_avg_sensitivity"]
scores_lle = data["quantus_local_lipschitz"]

@info "Converting dimensions..."
# Convert x_batch from NCHW (PyTorch) to WHCN (Flux/Metalhead)
# Permutation: 
# PyTorch dims: 0:N, 1:C, 2:H, 3:W
# Flux dims:    1:W, 2:H, 3:C, 4:N
# Map: 4, 3, 2, 1
x_jl = permutedims(x_py, (4, 3, 2, 1))

# Convert types
x_jl = Float32.(x_jl)

# Convert labels to 1-based indexing for Julia
y_jl = Int.(y_py) .+ 1
preds_jl_expected = Int.(preds_py) .+ 1

@info "Saving to JLD2..."
jldsave(OUTPUT_FILE; 
    x_batch=x_jl, 
    y_batch=y_jl,
    py_logits=logits_py,
    expected_preds=preds_jl_expected,
    quantus_avg_sensitivity=scores_as,
    quantus_local_lipschitz=scores_lle
)

@info "Sanity fixture created at $OUTPUT_FILE"
