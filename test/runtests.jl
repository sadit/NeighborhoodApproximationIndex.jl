# This file is a part of NeighborhoodApproximationIndex.jl

using NeighborhoodApproximationIndex
using Test

using KCenters, SimilaritySearch, NeighborhoodApproximationIndex

@testset "NeighborhoodApproximationIndex.jl" begin
    dim = 8
    n = 100_000
    X = randn(Float32, dim, n)
    Q = [randn(Float32, dim) for i in 1:100]
    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, X)
    kbuild, ksearch, t, recall = 8, 3, 1, 0.9
    P = Performance(seq, Q, 10)
    index = DeloneInvertedFile(dist, X; centers=:exhaustive, refs=:rand, k=kbuild, t=t, numcenters=2_000)
    #index = SearchGraph(; dist, db=MatrixDatabase(X))
    #index!(index)
    p = probe(P, copy(index, k=ksearch))
    # p = probe(P, index)
    @info "before optimization: $(index)" (recall=p.macrorecall, queries_per_second= 1 / p.searchtime, eval_ratio=p.evaluations / length(X))
    @test p.macrorecall > recall
    #optimize!(P, index, 0.9, verbose=true)
    #p = probe(P, index)

    #@info "after optimization: $(string(index))" (recall=p.macrorecall, queries_per_second= 1 / p.searchtime, eval_ratio=p.evaluations / length(X))
    # we can expect a small recall reduction since we are using a db's subset for optimizin the index
    
end
