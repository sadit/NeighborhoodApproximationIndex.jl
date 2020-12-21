using NeighborhoodApproximationIndex
using Documenter

makedocs(;
    modules=[NeighborhoodApproximationIndex],
    authors="Eric S. Tellez",
    repo="https://github.com/donsadit@gmail.com/NeighborhoodApproximationIndex.jl/blob/{commit}{path}#L{line}",
    sitename="NeighborhoodApproximationIndex.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://donsadit@gmail.com.github.io/NeighborhoodApproximationIndex.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/donsadit@gmail.com/NeighborhoodApproximationIndex.jl",
)
