# This file is a part of NeighborhoodApproximationIndex.jl

using Test, SimilaritySearch, NeighborhoodApproximationIndex

@testset "NeighborhoodApproximationIndex.jl" begin
    dim = 8
    n = 10^6
    m = 1000
    k = 30
    A = randn(Float32, dim, n)
    X = MatrixDatabase(A)
    Q = MatrixDatabase(randn(Float32, dim, m))
    centersrecall = 0.95
    numcenters = 10ceil(Int, sqrt(n))
    sample = unique(rand(1:n, 2numcenters))[1:numcenters]; sort!(sample)
    refs = MatrixDatabase(A[:, sample])
    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, X)
    kbuild, ksearch, t, minrecall = 3, 10, 1, 0.1
    parallel_block = 1000
    index = KnrIndex(dist, X; kbuild, ksearch, parallel_block, centersrecall, refs, rerank=true)
    @info "creating gold standard"
    Igold, Dgold, gsearchtime = timedsearchbatch(seq, Q, k; parallel=true)
    @info "searching in the index"
    Ires, Dres, tsearchtime = timedsearchbatch(index, Q, k; parallel=true)
    recall = macrorecall(Igold, Ires)
    @info "before optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $gsearchtime, index: $tsearchtime"
    @test recall > minrecall

    res = KnnResult(10)
    @time search(index, Q[1], res)
    res = reuse!(res)
    println("================ threads:", Threads.nthreads())
    @time search(index, Q[2], res)
    #optimize!(P, index, 0.9, verbose=true)
    #p = probe(P, index)

    #@info "after optimization: $(string(index))" (recall=p.macrorecall, queries_per_second= 1 / p.searchtime, eval_ratio=p.evaluations / length(X))
    # we can expect a small recall reduction since we are using a db's subset for optimizin the index
    
end
