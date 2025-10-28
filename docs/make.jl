using XAIMetrics
using Documenter

DocMeta.setdocmeta!(XAIMetrics, :DocTestSetup, :(using XAIMetrics); recursive=true)

makedocs(;
    modules=[XAIMetrics],
    authors="Janes Sanne <janes.luc.sanne@campus.tu-berlin.de>",
    sitename="XAIMetrics.jl",
    format=Documenter.HTML(;
        canonical="https://JeanAnNess.github.io/XAIMetrics.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JeanAnNess/XAIMetrics.jl",
    devbranch="master",
)
