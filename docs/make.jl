using XAIMetrics
using Documenter
using Literate

LITERATE_DIR = joinpath(@__DIR__, "src", "literate")
OUT_DIR = joinpath(@__DIR__, "src", "generated")

# Use Literate.jl to generate docs and notebooks of examples
function convert_literate(dir_in, dir_out)
    for p in readdir(dir_in)
        path = joinpath(dir_in, p)

        if isdir(path)
            convert_literate(path, joinpath(dir_out, p))
        else # isfile
            Literate.markdown(path, dir_out; documenter = true) # Markdown for Documenter.jl
            Literate.notebook(path, dir_out) # .ipynb notebook
            Literate.script(path, dir_out) # .jl script
        end
    end
    return nothing
end

isdir(LITERATE_DIR) && convert_literate(LITERATE_DIR, OUT_DIR)

DocMeta.setdocmeta!(XAIMetrics, :DocTestSetup, :(using XAIMetrics); recursive = true)

makedocs(;
    modules = [XAIMetrics],
    authors = "Janes Sanne <janes.luc.sanne@campus.tu-berlin.de>",
    sitename = "XAIMetrics.jl",
    format = Documenter.HTML(;
        canonical = "https://JeanAnNess.github.io/XAIMetrics.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "generated/getting_started.md",
        "API Reference" => [
            "Configurations" => "configurations.md",
            "Functions: Normalizations" => "normalizations.md",
            "Functions: Perturbations" => "perturbations.md",
            "Functions: Similarities" => "similarities.md",
        ],
        "Metrics" => [
            "Overview" => "metrics/_overview.md",
            "Axiomatic" => "metrics/axiomatic.md",
            "Complexity" => "metrics/complexity.md",
            "Faithfulness" => "metrics/faithfulness.md",
            "Localisation" => "metrics/localisation.md",
            "Randomisation" => "metrics/randomisation.md",
            "Robustness" => "metrics/robustness.md",
        ]
    ],
)

deploydocs(;
    repo = "github.com/JeanAnNess/XAIMetrics.jl",
    devbranch = "main",
    push_preview = true,
)
