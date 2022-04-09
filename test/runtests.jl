# This file is a part of NeighborhoodApproximationIndex.jl

using Test, SimilaritySearch, NeighborhoodApproximationIndex, LinearAlgebra, UnicodePlots

function runtest(; dim, n, m,
    numcenters=3ceil(Int, sqrt(n)), k=10, centersrecall=0.95, kbuild=1, ksearch=1,
    parallel_block=1000, ordering=DistanceOrdering(), minrecall=1.0, initial=:dnet, maxiters=0)
    A = randn(Float32, dim, n)
    B = randn(Float32, dim, m)
    X = MatrixDatabase(A)
    Q = MatrixDatabase(B)
    # sample = unique(rand(1:n, 2numcenters))[1:numcenters]; sort!(sample)
    # refs = MatrixDatabase(A[:, sample])
    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, X)
    @info "creating gold standard"
    @time Igold, Dgold, gsearchtime = timedsearchbatch(seq, Q, k; parallel=true)
    @info "creating the KnrIndex"
    indextime = @elapsed index = KnrIndex(dist, X; kbuild, ksearch, parallel_block, centersrecall, initial, maxiters, ordering)
    @test length(index) == length(X)
    @info "searching in the index"
    @show dim, n, m , numcenters, k, centersrecall, ordering
    @time Ires, Dres, tsearchtime = timedsearchbatch(index, Q, k; parallel=true)
    recall = macrorecall(Igold, Ires)
    @info "before optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), index: $(tsearchtime * m), index-construction: $indextime"
    
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
    @test recall >= min(0.5, minrecall)

    @info "**** optimizing MinRecall(0.95) ****"
    opttime = @elapsed optimize!(index, MinRecall(0.95); verbose=false)
    @time Ires, Dres, tsearchtime = timedsearchbatch(index, Q, k; parallel=true)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), index: $(tsearchtime * m), optimization-time: $opttime"
    @test recall >= min(0.7, minrecall)
    
    @info "********************* generic searches *******************"
    res = KnnResult(10)
    @time search(index, Q[1], res)
    res = reuse!(res)
    @time search(index, Q[2], res)
    F = sort!(length.(index.invfile.lists), rev=true)
    println(histogram(F, nbins=20))
    println(lineplot(F))
end

@testset "NeighborhoodApproximationIndex.jl" begin
    @info "********************* JIT warming *********************"  # for fast benchmarking
    centersrecall = 0.95
    runtest(; dim=2, n=10^3, m=10, k=10, centersrecall, kbuild=1, ksearch=1, ordering=ordering=InternalDistanceOrdering(), minrecall=0)
    runtest(; dim=2, n=10^3, m=10, k=10, centersrecall, kbuild=1, ksearch=1, ordering=DistanceOnTopKOrdering(10), minrecall=0)
    runtest(; dim=2, n=10^3, m=10, k=10, centersrecall, kbuild=1, ksearch=1, minrecall=0)
    # DistanceOnTopKOrdering and InternalDistanceOrdering need high `kbuild` and `ksearch` to generate
    # some discrimination power by the internal distance
    # useful for costly distance functions or whenever the dataset is pretty large
    runtest(; dim=4, n=10^5, m=1000, numcenters=10, k=100, centersrecall, kbuild=5, ksearch=5, ordering=InternalDistanceOrdering(), minrecall=0)
    @info "********************* Real search (top-k) *********************"
    # useful for costly distance functions
    runtest(; dim=4, n=10^5, m=1000, k=100, centersrecall, kbuild=5, ksearch=5, ordering=DistanceOnTopKOrdering(1000), minrecall=0.6)
    @info "********************* Real search *********************"
    # most usages
    runtest(; dim=4, n=10^6, m=1000, k=100, centersrecall, kbuild=1, ksearch=1, ordering=DistanceOrdering())
end
