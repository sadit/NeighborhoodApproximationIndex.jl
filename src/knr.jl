# This file is a part of NeighborhoodApproximationIndex.jl

import SimilaritySearch: search, getpools, getknnresult, index!
using InvertedFiles, Intersections, KCenters, StatsBase, Parameters, LinearAlgebra
export KnrIndex, search

mutable struct KnrOpt
    ksearch::Int32
end

struct KnrIndex{
            DistType<:SemiMetric,
            DataType<:AbstractDatabase,
            CentersIndex<:AbstractSearchContext,
            InvIndexType<:AbstractInvertedFile
        } <: AbstractSearchContext
    dist::DistType
    db::DataType
    centers::CentersIndex
    invfile::InvIndexType
    kbuild::Int32
    rerank::Bool
    opt::KnrOpt
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

function Base.push!(idx::KnrIndex, obj; pools=getpools(idx), encpools=getpools(idx.centers))
    res = getencodeknnresult(idx.kbuild, pools)
    search(idx.centers, obj, res; pools=encpools)
    push!(idx.invfile, res)
    idx
end

get_parallel_block(n) = min(n, 8 * Threads.nthreads())

function Base.append!(idx::KnrIndex, db;
        parallel_block=get_parallel_block(length(db)),
        pools=getpools(idx),
        verbose=true
    )

    append!(idx.db, db)
    index!(idx; parallel_block, pools, verbose)
end

function index!(idx::KnrIndex; parallel_block=get_parallel_block(length(idx.db)), pools=nothing, verbose=true)
    sp = length(idx) + 1
    n = length(idx.db)
    E = [KnnResult(idx.kbuild) for _ in 1:parallel_block]
    while sp < n
        ep = min(n, sp + parallel_block - 1)
        verbose && println(stderr, "$(typeof(idx)) appending chunk ", (sp=sp, ep=ep, n=n), " ", Dates.now())
    
        Threads.@threads for i in sp:ep
            begin
                res = reuse!(E[i - sp + 1], idx.kbuild)
                search(idx.centers, idx[i], res)
            end
        end

        append!(idx.invfile, VectorDatabase(E), ep-sp+1; parallel_block)
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
        ksearch=1,
        centersrecall::AbstractFloat=length(db) > 10^3 ? 0.95 : 1.0,
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
            centers = SearchGraph(; db=refs, dist, verbose)
            index!(centers; parallel_block)
            optimize!(centers, OptimizeParameters(kind=MinRecall(centersrecall)))
        end
    end
    
    invfile = invfiletype(length(centers), invfiledist, Int32)
    idx = KnrIndex(dist, db, centers, invfile, kbuild, rerank, KnrOpt(ksearch))
    pools = pools === nothing ? getpools(idx) : pools
    index!(idx; parallel_block, pools, verbose)
    idx
end
