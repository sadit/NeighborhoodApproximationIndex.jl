# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

using Test


@testset "Testing Delone Inverted Index" begin
    using KCenters, SimilaritySearch
    using NeighborhoodApproximationIndex

    dim = 4
    n = 10000
    X = [randn(dim) for i in 1:n]
    Q = [randn(dim) for i in 1:100]
    dist = SqL2Distance()
    ksearch = 5
    seq = ExhaustiveSearch(dist, X)
    P = Performance(seq, Q, ksearch)
    index = DeloneInvIndex(dist, X; initial=:rand, ksearch=3)
    println(stderr, ([length(lst) for lst in index.lists]))
    p = probe(P, index)
    @info "before optimization: $(string(index))" (recall=p.macrorecall, queries_per_second= 1 / p.searchtime, eval_ratio=p.evaluations / length(X))
    optimize!(P, index, 0.9, verbose=true)
    p = probe(P, index)

    @info "after optimization: $(string(index))" (recall=p.macrorecall, queries_per_second= 1 / p.searchtime, eval_ratio=p.evaluations / length(X))
    # we can expect a small recall reduction since we are using a db's subset for optimizin the index
    @test p.macrorecall > 0.7
end