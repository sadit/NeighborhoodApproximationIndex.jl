# This file is a part of NeighborhoodApproximationIndex.jl
import SimilaritySearch: search, getpools, getknnresult, index!
using InvertedFiles, Intersections, KCenters, StatsBase, Parameters, LinearAlgebra
export KnrIndex, search

struct KnrIndex{
            DistType<:SemiMetric,
            DataType<:AbstractDatabase,
            InvertedFileType<:InvertedFiles.AbstractInvertedFile,
            CentersType<:AbstractSearchContext
        } <: AbstractSearchContext
    dist::DistType
    db::DataType
    centers::CentersType
    invfile::InvertedFileType
    kbuild::Int32
    ksearch::Int32
    rerank::Bool
end

@inline Base.length(idx::KnrIndex) = length(idx.invfile)
Base.show(io::IO, idx::KnrIndex) = print(io, "{$(typeof(idx)) centers=$(typeof(idx.centers)), n=$(length(idx))}")

const GlobalEncodeKnnResult = [KnnResult(10)]

function __init__()
    for i in 2:Threads.nthreads()
        push!(GlobalEncodeKnnResult, KnnResult(10))
    end
end

struct KnrPools
    results::Vector{KnnResult}
    encoderesults::Vector{KnnResult}
end

@inline function getknnresult(k::Integer, pools::KnrPools)
    res = @inbounds pools.results[Threads.threadid()]
    reuse!(res, k)
end

@inline function getencodeknnresult(k::Integer, pools::KnrPools)
    res = @inbounds pools.encoderesults[Threads.threadid()]
    reuse!(res, k)
end

getpools(::KnrIndex; results=SimilaritySearch.GlobalKnnResult, encoderesults=GlobalEncodeKnnResult) =
    KnrPools(results, encoderesults)

function Base.push!(D::KnrIndex, obj; pools=getpools(D), encpools=getpools(D.centers))
    res = getencodeknnresult(D.kbuild, pools)
    search(D.centers, obj, res; pools=encpools)
    push!(D.invfile, (length(D) + 1) => zip(res.id, res.dist))
    D
end

get_parallel_block(n) = min(n, 8 * Threads.nthreads())

function Base.append!(D::KnrIndex, db;
        parallel_block=get_parallel_block(length(db)),
        pools=getpools(D),
        verbose=true
    )

    append!(D.db, db)
    index!(D; parallel_block, pools, verbose)
end

function index!(D::KnrIndex;
        parallel_block=get_parallel_block(length(D.db)),
        pools=nothing,
        verbose=true
    )

    sp = length(D) + 1
    n = length(D.db)
    E = [KnnResult(D.kbuild) for _ in 1:parallel_block]
    while sp < n
        ep = min(n, sp + parallel_block - 1)
        verbose && println(stderr, "$(typeof(D)) appending chunk ", (sp=sp, ep=ep, n=n), " ", Dates.now())
    
        Threads.@threads for i in sp:ep
            begin
                res = reuse!(E[i - sp + 1], D.kbuild)
                search(D.centers, D[i], res)
            end
        end

        append!(D.invfile, VectorDatabase(E), ep-sp+1; parallel_block)
        sp = ep + 1
    end
end

"""
    KnrIndex(
        dist::SemiMetric,
        db;
        numcenters=ceil(Int, sqrt(length(db))),
        refs=:rand,
        centers=:delone,
        k=5,
        pools=nothing,
        parallel_block=Threads.nthreads() > 1 ? 1024 : 1,
        verbose=true
    )

Creates a `KnrIndex`, high level interface.

- `dist`: Distance object (a `SemiMetric` object, see `Distances.jl`)
- `db`: Objects to be indexed
  Please note that these indexes will be created with generic parameters
- `refs`: the se of reference, only used if `centers=nothing`
- `pools`: an object with preallocated caches specific for `KnrIndex`, if `pools=nothing` it will use default caches.
- `parallel_block` Parallel construction works on batches, this is the size of these blocks
- `verbose` true if you want to see messages
"""
function KnrIndex(
        dist::SemiMetric,
        db::AbstractDatabase;
        invfiletype=BinaryInvertedFile,
        invfiledist=JaccardDistance(),
        refs=references(dist, db),
        centers=nothing,
        kbuild=3,
        ksearch=kbuild,
        centersrecall::AbstractFloat=1.0,
        rerank=true,
        pools=nothing,
        parallel_block=get_parallel_block(length(db)),
        verbose=false
    )

    kbuild = convert(Int32, kbuild)
    ksearch = convert(Int32, ksearch)

    if centers === nothing
        if centersrecall == 1.0
            centers = ExhaustiveSearch(; db=refs, dist)
        else
            0 < centersrecall < 1 || throw(ArgumentError("the expected recall for centers index should be 0 < centersrecall < 0"))
            centers = SearchGraph(; db=refs, dist)
            index!(centers; parallel_block)
            optimize!(centers, OptimizeParameters(kind=MinRecall(centersrecall)))
        end
    end
    
    invfile = invfiletype(length(centers), invfiledist, Int32)
    D = KnrIndex(dist, db, centers, invfile, kbuild, ksearch, rerank)
    pools = pools === nothing ? getpools(D) : pools
    index!(D; parallel_block, pools, verbose)
    D
end

"""
    search(idx::KnrIndex, q::T, res::KnnResult; rerank=idx.rerank, pools=getpools(D))

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(idx::KnrIndex, q, res::KnnResult; ksearch=idx.ksearch, rerank=idx.rerank, pools=getpools(idx))
    enc = getencodeknnresult(ksearch, pools)
    search(idx.centers, q, enc)
    Q = prepare_posting_lists_for_querying(idx.invfile, enc)

    if rerank
        search(idx.invfile, Q) do objID, d  # it is also possible to add an additional state of filtering if the distance function is too costly
            @inbounds push!(res, objID, evaluate(idx.dist, q, idx[objID]))
        end
    else
        search(idx.invfile, Q) do objID, d
            push!(res, objID, d)
        end
    end

    (res=res, cost=0)
end

#=
"""
    optimize!(perf::Performance, index::KnrIndex, recall=0.9; numqueries=128, verbose=false, k=10)

Tries to configure `index` to achieve the specified recall for fetching `k` nearest neighbors.
"""
function optimize!(perf::Performance, index::KnrIndex, recall=0.9; numqueries=128, verbose=false, k=10)
    verbose && println("$(string(index)) optimizing for recall=$(recall)")
    index.opts.ksearch = 1
    p = probe(perf, index)

    while p.macrorecall < recall && index.opts.ksearch < length(index.lists)
        index.opts.ksearch += 1
        verbose && println(stderr, "$(string(index)) optimize! step ksearch=$(index.opts.ksearch), performance ", p)
        p = probe(perf, index)
    end

    verbose && println("$(string(index)) reached performance ", p)
    index
end=#
