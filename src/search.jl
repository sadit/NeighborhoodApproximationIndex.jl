# This file is a part of NeighborhoodApproximationIndex.jl

"""
    search(idx::KnrIndex, q, res::KnnResult; ksearch=idx.ksearch, rerank=idx.rerank, pools=getpools(D))

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(idx::KnrIndex, q, res::KnnResult;
        ksearch=idx.opt.ksearch, rerank=idx.rerank, pools=getpools(idx))

    enc = getencodeknnresult(ksearch, pools)
    search(idx.centers, q, enc)
    Q = prepare_posting_lists_for_querying(idx.invfile, enc)

    cost = 0
    if rerank
        search(idx.invfile, Q) do objID, d  # it is also possible to add an additional state of filtering if the distance function is too costly
            @inbounds push!(res, objID, evaluate(idx.dist, q, idx[objID]))
            cost += 1
        end
    else
        search(idx.invfile, Q) do objID, d
            push!(res, objID, d)
        end
    end

    (res=res, cost=cost)
end