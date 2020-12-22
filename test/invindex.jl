# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Test


@testset "Testing Delone Inverted Index" begin
    using KCenters, SimilaritySearch
    using NeighborhoodApproximationIndex
    using StatsBase: mean

    dim = 4
    n = 10000
    X = [randn(dim) for i in 1:n]
    Q = [randn(dim) for i in 1:100]
    dist = l2_distance
    ksearch = 5
    P = Performance(dist, X, Q, expected_k=ksearch)
    
    index = fit(DeloneInvIndex, dist, X; numcenters=ceil(Int, sqrt(n)), initial=:dnet, region_expansion=3)
    println(stderr, ([length(lst) for lst in index.lists]))
    p = probe(P, index, dist)
    @info "before optimization" (recall=p.recall, speedup=p.exhaustive_search_seconds / p.seconds, eval_ratio=p.evaluations / length(X))
    optimize!(index, dist, 0.9, verbose=true)
    p = probe(P, index, dist)
    @info "after optimization" (recall=p.recall, speedup=p.exhaustive_search_seconds / p.seconds, eval_ratio=p.evaluations / length(X))
    # we can expect a small recall reduction since we are using a db's subset for optimizin the index
    @test p.recall > 0.7
end