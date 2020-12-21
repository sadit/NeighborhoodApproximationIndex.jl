using NeighborhoodApproximationIndex
using Documenter

makedocs(;
    modules=[NeighborhoodApproximationIndex],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/NeighborhoodApproximationIndex.jl/blob/{commit}{path}#L{line}",
    sitename="NeighborhoodApproximationIndex.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sadit.github.io/NeighborhoodApproximationIndex.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sadit/NeighborhoodApproximationIndex.jl",
)
