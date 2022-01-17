# This file is a part of NeighborhoodApproximationIndex.jl
import SimilaritySearch: search
using InvertedFiles, Intersections, KCenters, StatsBase, Parameters, LinearAlgebra
export DeloneInvertedFile, search

const GlobalEncodeKnnResult = [KnnResult(10)]
const GlobalEncodeVector = [SVEC()]

function __init__()
    for i in 2:Threads.nthreads()
        push!(GlobalEncodeKnnResult, KnnResult(10))
        push!(GlobalEncodeVector, SVEC())
    end
end

@inline getencodeknnresult() = @inbounds GlobalEncodeKnnResult[Threads.threadid()]
@inline getencodevector() = @inbounds GlobalEncodeVector[Threads.threadid()]

@with_kw struct DeloneInvertedFile{
        DistType<:SemiMetric,
        DataType<:AbstractDatabase,
        InvertedFileType<:InvertedFile,
        CentersType<:AbstractSearchContext
        } <: AbstractSearchContext
    dist::DistType
    db::DataType
    centers::CentersType
    invfile::InvertedFileType
    k::Int32 = 7
    t::Int32 = 1
end

@inline Base.getindex(D::DeloneInvertedFile, i) = @inbounds D.db[i]
@inline Base.length(D::DeloneInvertedFile) = D.invfile.n
@inline Base.eachindex(D::DeloneInvertedFile) = 1:length(D)

Base.copy(D::DeloneInvertedFile;
    dist=D.dist,
    db=D.db,
    centers=D.centers,
    invfile=D.invfile,
    k=D.k,
    t=D.t
    ) = DeloneInvertedFile(; dist, db, centers, invfile, k, t)

function encode!(D::DeloneInvertedFile, obj::T, v::DVEC) where T
    empty!(v)
    encres = getencodeknnresult()
    reuse!(encres, D.k)

    for (id_, d_) in search(D.centers, obj, encres).res
        v[id_] = d_
    end

    normalize!(v)
end

function Base.push!(D::DeloneInvertedFile, obj, v::DVEC; push_item=true)
    push_item && push!(D.db, obj)
    push!(D.invfile, length(D)+1 => v)
end

function Base.append!(D::DeloneInvertedFile, db; parallel_block=1, encode=true, verbose=true)
    append!(D.db, db)
    index!(D; parallel_block, encode, verbose)
end

function _parallel_index!(D::DeloneInvertedFile, parallel_block, verbose)
    nt = Threads.nthreads()
    sp = length(D)+1
    n = length(D.db)
    venc = [SVEC() for i in 1:parallel_block]
    while sp < n
        ep = min(n, sp + parallel_block - 1)
        verbose && println(stderr, "$(typeof(D)) appending chunk ", (sp=sp, ep=ep, n=n), " ", Dates.now())
        
        Threads.@threads for i in sp:ep
            @inbounds encode!(D, D[i], venc[i-sp+1])
        end

        for i in sp:ep
            @inbounds push!(D, D[i], venc[i-sp+1]; push_item=false)
        end

        sp = ep + 1
    end
end

function SimilaritySearch.index!(D::DeloneInvertedFile; parallel_block=1, encode=true, verbose=true)
    encode = encode || !(index.db[1] isa DVEC)

    if !encode
        for i in length(D)+1:length(D.db)
            v = D[i]
            push!(D, v, v; push_item=false)
        end
        _parallel_index!(D, parallel_block, verbose)
    else
        for i in length(D)+1:length(D.db)
            v = D[i]
            venc = encode!(D, v, getencodevector())
            push!(D, v, venc; push_item=false)
        end
    end

    D
end

"""
    DeloneInvertedFile(
        dist::SemiMetric,
        db;
        numcenters=ceil(Int, sqrt(length(db))),
        refs=:rand,
        centers=:delone,
        k=5,
        t=1,
        parallel_block=Threads.nthreads() > 1 ? 1024 : 1,
        verbose=true
    )

Creates a `DeloneInvertedFile`, high level interface.

- `dist`: Distance object (a `SemiMetric` object, see `Distances.jl`)
- `db`: Objects to be indexed
- `centers`: An index on a set of references used to create the index. It can also be a symbol that describes how to create the index. Valid symbols are:
  - `:delone` creates a DeloneInvertedFile
  - `:graph` creates a SearchGraph
  - `:exhaustive` creates an ExhaustiveSearch
  Please note that these indexes will be created with generic parameters
- `numcenters` number of centers to insert, only used if `centers` is not an index
- `refs`: References, it can be raw objects or symbols indicating how to compute them `:rand`, `:dnet`, `:fft` (see `KCenters.jl`). Only used if `centers` is not an index
- `k` the number of references to be used
- `t` the number of occurrences in the inverted file required to accept an object as candidate in the result set
- `parallel_block` Parallel construction works on batches, this is the size of these blocks
- `verbose` true if you want to see messages
"""
function DeloneInvertedFile(
        dist::SemiMetric,
        db;
        numcenters=ceil(Int, sqrt(length(db))),
        refs=:rand,
        centers=:delone,
        k=5,
        t=1,
        parallel_block=Threads.nthreads() > 1 ? 1024 : 1,
        verbose=true
    )
    db = convert(AbstractDatabase, db)

    if centers === nothing || centers in (:delone, :graph, :exhaustive)
        if refs === :rand
            refs = SubDatabase(db, unique(rand(1:length(db), numcenters)))
        elseif refs in (:dnet, :fft)
            train = SubDatabase(db, unique(rand(1:length(db), 3*numcenters)))
            C = kcenters(dist, train, numcenters; initial=refs, maxiters=0)
            refs = C.centers[C.dmax .> 0.0]
        end
        
        centers = let refs = convert(AbstractDatabase, refs)
            if centers === :delone
                m = ceil(Int, length(refs) / log2(length(refs)))
                DeloneInvertedFile(dist, refs; numcenters=m, centers=:exhaustive, k=3, t=1, verbose)
            elseif centers === :exhaustive
                ExhaustiveSearch(dist, refs)
            elseif centers === :graph
                centers = SearchGraph(; dist, db=refs)
                index!(centers; parallel_block=64)
            end
        end
    end
    
    D = DeloneInvertedFile(;
        dist, k, t, db, centers,
        invfile=InvertedFile(length(centers)),
    )

    index!(D; parallel_block, verbose)
    D
end

"""
    search(D::DeloneInvertedFile, q::T, res::KnnResult)

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(D::DeloneInvertedFile, q, res::KnnResult; t=D.t)
    Q = prepare_posting_lists_for_querying(D.invfile, encode!(D, q, getencodevector()))
    count = Ref(0)
	umerge(Q, t) do L, P, m
        @inbounds begin
            id = L[1].I[P[1]]
		    push!(res, id, evaluate(D.dist, D[id], q))
            count[] += 1
        end
	end

    (res=res, cost=count[])
end

#=
"""
    optimize!(perf::Performance, index::DeloneInvertedFile, recall=0.9; numqueries=128, verbose=false, k=10)

Tries to configure `index` to achieve the specified recall for fetching `k` nearest neighbors.
"""
function optimize!(perf::Performance, index::DeloneInvertedFile, recall=0.9; numqueries=128, verbose=false, k=10)
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
