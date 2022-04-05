using Documenter, NeighborhoodApproximationIndex

makedocs(;
    modules=[NeighborhoodApproximationIndex],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/NeighborhoodApproximationIndex.jl/blob/{commit}{path}#L{line}",
    sitename="NeighborhoodApproximationIndex.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://sadit.github.io/NeighborhoodApproximationIndex.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md"
    ],
)

deploydocs(;
    repo="github.com/sadit/NeighborhoodApproximationIndex.jl",
    devbranch=nothing,
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"]
)
