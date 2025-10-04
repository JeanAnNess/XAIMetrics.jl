using ExplanationMetrics
using Documenter

DocMeta.setdocmeta!(ExplanationMetrics, :DocTestSetup, :(using ExplanationMetrics); recursive=true)

makedocs(;
    modules=[ExplanationMetrics],
    authors="Janes Sanne <janes.luc.sanne@campus.tu-berlin.de>",
    sitename="ExplanationMetrics.jl",
    format=Documenter.HTML(;
        canonical="https://JeanAnNess.github.io/ExplanationMetrics.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JeanAnNess/ExplanationMetrics.jl",
    devbranch="master",
)
