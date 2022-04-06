# This file is a part of NeighborhoodApproximationIndex.jl

"""
    search(idx::KnrIndex, q, res::KnnResult; ksearch=idx.opt.ksearch, ordering=idx.ordering, pools=getpools(D))

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(idx::KnrIndex, q, res::KnnResult; pools=getpools(idx), ksearch=idx.opt.ksearch)
    enc = getencodeknnresult(ksearch, pools)
    search(idx.centers, q, enc)
    Q = prepare_posting_lists_for_querying(idx.invfile, enc)
    search_(idx, q, enc, Q, res, idx.ordering)
end

function search_(idx::KnrIndex, q, _, Q, res::KnnResult, ::DistanceOrdering)
    cost = 0
    umerge(Q) do L, P, _
        @inbounds objID = L[1][P[1]]
        @inbounds push!(res, objID, evaluate(idx.dist, q, idx[objID]))
        cost += 1
    end

    (res=res, cost=cost)
end

function search_(idx::KnrIndex, q, enc, Q, res::KnnResult, ordering::DistanceOnTopKOrdering)
    enc = reuse!(enc, ordering.top)
    search(idx.invfile, Q) do objID, d
        @inbounds push!(enc, objID, d) #evaluate(idx.dist, q, idx[objID]))
    end

    for (objID, d) in enc
        @inbounds push!(res, objID, evaluate(idx.dist, q, idx[objID]))
    end

    (res=res, cost=length(enc))
end

function search_(idx::KnrIndex, q, _, Q, res::KnnResult, ::InternalDistanceOrdering)
    search(idx.invfile, Q) do objID, d
        @inbounds push!(res, objID, d)
    end

    (res=res, cost=0)
end
