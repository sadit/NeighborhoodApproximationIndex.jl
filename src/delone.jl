# This file is a part of NeighborhoodApproximationIndex.jl
import SimilaritySearch: search
using InvertedFiles, Intersections, KCenters, StatsBase, Parameters, LinearAlgebra
export DeloneInvertedFile, search

@with_kw struct DeloneInvertedFile{
        DistType<:PreMetric,
        DataType<:AbstractVector,
        InvertedFileType<:InvertedFile,
        CentersType<:AbstractSearchContext,
        KnnResultType
        } <: AbstractSearchContext
    dist::DistType = SqL2Distance()
    db::DataType
    centers::CentersType
    invfile::InvertedFileType = InvertedFile()
    k::Int32 = 7
    t::Int32 = 1
    res::KnnResultType = KnnResult(1)
    v::DVEC = SVEC()
end

@inline Base.getindex(D::DeloneInvertedFile, i) = @inbounds D.db[i]

Base.copy(D::DeloneInvertedFile;
    dist=D.dist,
    db=D.db,
    centers=D.centers,
    invfile=D.invfile,
    k=D.k,
    t=D.t,
    res=KnnResult(1),
    v=SVEC()
    ) = DeloneInvertedFile(; dist, db, centers, invfile, k, t, res, v)

function encode!(D::DeloneInvertedFile, obj::T) where T
    empty!(D.v)
    empty!(D.res, D.k)

    for (id_, d_) in search(D.centers, obj, D.res)
        D.v[id_] = d_
    end

    normalize!(D.v)
end

function Base.push!(D::DeloneInvertedFile, obj, v::DVEC)
    push!(D.db, obj)
    push!(D.invfile, length(D.db) => v)
end

function Base.append!(D::DeloneInvertedFile, db; parallel_block=1, encode=true, verbose=true)
    encode = encode || !(db[1] isa DVEC)

    if !encode
        for v in db
            push!(D, v, v)
        end
    elseif parallel_block > 1
        nt = Threads.nthreads()
        D_ = [copy(D) for i in 1:nt]
        sp = 1
        n = length(db)
        
        while sp < n
            ep = min(n, sp + parallel_block)
            verbose && println(stderr, "$(typeof(D)) appending chunk ", (sp=sp, ep=ep, n=n), " ", Dates.now())
            begin
                Threads.@threads for i in sp:ep
                    encode!(D_[Threads.threadid()], db[i])
                end
        
                for i in eachindex(D_)
                    push!(D, db[sp+i-1], D_[i].v)
                end
            end

            sp = ep + 1
        end
    else
        for v in db
            push!(D, v, encode!(D, v))
        end
    end

    D
end


function DeloneInvertedFile(
        dist::PreMetric,
        db;
        numcenters=ceil(Int, sqrt(length(db))),
        train=rand(db, 2*numcenters),
        k=5,
        t=1,
        initial=:dnet,
        maxiters=3,
        parallel_block=Threads.nthreads() > 1 ? 1024 : 1,
        verbose=true
    )

    C = kcenters(dist, train, numcenters; initial=initial, maxiters=maxiters)
    D = DeloneInvertedFile(;
        dist=dist,
        db=eltype(db)[],
        centers=ExhaustiveSearch(dist, C.centers),
        k=k,
        t=t
    )

    append!(D, db; parallel_block, verbose)
    D
end

"""
    search(D::DeloneInvertedFile, q::T, res::KnnResult)

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(D::DeloneInvertedFile, q, res::KnnResult; t=D.t)
    Q = prepare_posting_lists_for_querying(D.invfile, encode!(D, q))

	umerge(Q, t) do L, P, m
        @inbounds begin
            id = L[1].I[P[1]]
		    push!(res, id, evaluate(D.dist, D.db[id], q))
        end
	end

    res
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
