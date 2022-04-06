# This file is a part of NeighborhoodApproximationIndex.jl

using Test, SimilaritySearch, NeighborhoodApproximationIndex

function runtest(; dim, n, m,
    numcenters=10ceil(Int, sqrt(n)), k=10, centersrecall=0.95, kbuild=1, ksearch=1, minrecall=0.1, parallel_block=1000)
    A = randn(Float32, dim, n)
    X = MatrixDatabase(A)
    Q = MatrixDatabase(randn(Float32, dim, m))
    sample = unique(rand(1:n, 2numcenters))[1:numcenters]; sort!(sample)
    refs = MatrixDatabase(A[:, sample])
    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, X)
    @info "creating gold standard"
    @time Igold, Dgold, gsearchtime = timedsearchbatch(seq, Q, k; parallel=true)
    indextime = @elapsed index = KnrIndex(dist, X; kbuild, ksearch, parallel_block, centersrecall, refs, rerank=true)
    @test length(index) == length(X)
    @info "searching in the index"
    @time Ires, Dres, tsearchtime = timedsearchbatch(index, Q, k; parallel=true)
    recall = macrorecall(Igold, Ires)
    @info "before optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), index: $(tsearchtime * m), index-construction: $indextime"
    @test recall > minrecall
    res = KnnResult(10)
    @time search(index, Q[1], res)
    res = reuse!(res)
    println("================ threads:", Threads.nthreads())
    @time search(index, Q[2], res)

    @info "**** optimizing ParetoRadius() ****"
    opttime = @elapsed optimize!(index, ParetoRadius(); verbose=false)
    @time Ires, Dres, tsearchtime = timedsearchbatch(index, Q, k; parallel=true)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), index: $(tsearchtime * m), optimization-time: $opttime"

    @info "**** optimizing ParetoRecall() ****"
    opttime = @elapsed optimize!(index, ParetoRecall(); verbose=false)
    @time Ires, Dres, tsearchtime = timedsearchbatch(index, Q, k; parallel=true)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), index: $(tsearchtime * m), optimization-time: $opttime"

    @info "**** optimizing MinRecall(0.95) ****"
    opttime = @elapsed optimize!(index, MinRecall(0.95); verbose=false)
    @time Ires, Dres, tsearchtime = timedsearchbatch(index, Q, k; parallel=true)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), index: $(tsearchtime * m), optimization-time: $opttime"

end

@testset "NeighborhoodApproximationIndex.jl" begin
    @info "********************* JIT warming *********************"
    centersrecall = 0.95
    runtest(; dim=2, n=100, m=10, numcenters=10, k=3, centersrecall)
    @info "********************* Real search *********************"
    runtest(; dim=8, n=10^5, m=1000, k=10, centersrecall, kbuild=3, ksearch=1)
    #optimize!(P, index, 0.9, verbose=true)
    #p = probe(P, index)

    #@info "after optimization: $(string(index))" (recall=p.macrorecall, queries_per_second= 1 / p.searchtime, eval_ratio=p.evaluations / length(X))
    # we can expect a small recall reduction since we are using a db's subset for optimizin the index
end
