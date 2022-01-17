# This file is a part of NeighborhoodApproximationIndex.jl

using NeighborhoodApproximationIndex
using Test

using KCenters, SimilaritySearch, NeighborhoodApproximationIndex

@testset "NeighborhoodApproximationIndex.jl" begin
    dim = 8
    n = 100_000
    m = 1000
    numcenters = 10_000
    k = 10
    X = MatrixDatabase(randn(Float32, dim, n))
    Q = MatrixDatabase(randn(Float32, dim, m))
    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, X)
    kbuild, ksearch, t, minrecall = 3, 2, 2, 0.9
    index = DeloneInvertedFile(dist, X; centers=:graph, refs=:rand, k=kbuild, t=t, numcenters=numcenters)
    Igold, Dgold, searchtime = timedsearchbatch(seq, Q, k)
    Ires, Dres, searchtime = timedsearchbatch(index, Q, k)
    recall = macrorecall(Igold, Ires)
    @info "before optimization: $(index)" (recall=recall, qps=queries_per_second=1/searchtime)
    @test recall > minrecall
    #optimize!(P, index, 0.9, verbose=true)
    #p = probe(P, index)

    #@info "after optimization: $(string(index))" (recall=p.macrorecall, queries_per_second= 1 / p.searchtime, eval_ratio=p.evaluations / length(X))
    # we can expect a small recall reduction since we are using a db's subset for optimizin the index
    
end
