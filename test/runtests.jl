# This file is a part of NeighborhoodApproximationIndex.jl

using NeighborhoodApproximationIndex
using Test

using KCenters, SimilaritySearch, NeighborhoodApproximationIndex


#include("invindex.jl")
#include("testknr.jl")

@testset "NeighborhoodApproximationIndex.jl" begin
    dim = 4
    n = 100_000
    X = [randn(dim) for i in 1:n]
    Q = [randn(dim) for i in 1:100]
    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, X)
    ksearch = 2
    P = Performance(seq, Q, 7)
    index = DeloneInvertedFile(dist, X; initial=:rand, k=ksearch)
    p = probe(P, index)
    @info "before optimization: $(index)" (recall=p.macrorecall, queries_per_second= 1 / p.searchtime, eval_ratio=p.evaluations / length(X))
    @test p.macrorecall > 0.7
    #optimize!(P, index, 0.9, verbose=true)
    #p = probe(P, index)

    #@info "after optimization: $(string(index))" (recall=p.macrorecall, queries_per_second= 1 / p.searchtime, eval_ratio=p.evaluations / length(X))
    # we can expect a small recall reduction since we are using a db's subset for optimizin the index
    
end
