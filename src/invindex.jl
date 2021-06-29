# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import SimilaritySearch: search, optimize!
using StatsBase
using KCenters

export DeloneInvIndex, search, optimize!

mutable struct DeloneInvIndexOptions
    n::Int32
    ksearch::Int32
end

struct DeloneInvIndex{DistType<:PreMetric, DataType<:AbstractVector, CentersType<:AbstractSearchContext} <: AbstractSearchContext
    dist::DistType
    db::DataType
    centers::CentersType
    lists::Vector{Vector{UInt32}}
    dmax::Vector{Float32}
    res::KnnResult
    opts::DeloneInvIndexOptions
end

Base.copy(I::DeloneInvIndex; dist=I.dist, db=I.db, centers=I.centers, lists=I.lists, dmax=I.dmax, res=I.res, opts=I.opts) =
    DeloneInvIndex(dist, db, centers, lists, dmax, res, opts)

Base.string(I::DeloneInvIndexOptions) = "{n=$(I.n), ksearch=$(I.ksearch)}"
Base.string(I::DeloneInvIndex) = "{DeloneInvIndex: dist=$(I.dist), refs=$(length(I.centers.db)), opts=$(string(I.opts))}"

"""
    DeloneInvIndex(dist::PreMetric, X::AbstractVector, kcenters_::ClusteringData; ksearch=3, k=10)
    DeloneInvIndex(dist::PreMetric, X::AbstractVector; numcenters=128, ksearch=3, initial=:dnet, maxiters=7, k=10)

Construct an approximate similarity search index based on a Delone partition on `X` using distance function `dist` using the initial set of points.
The `region_expasion` parameter indicates how queries are solved (looking for nearest regions). The parameter 
`initial` can be also a clustering strategy, and therefore, it selects `numcenters` centers as prototypes.

The supported values for `initial` are the following values.
    - `:fft` the _farthest first traversal_ selects a set of farthest points among them to serve as cluster seeds.
    - `:dnet` the _density net_ algorithm selects a set of points following the same distribution of the datasets; in contrast with a random selection, `:dnet` ensures that the selected points are not ``\\lfloor n/k \\rfloor`` nearest neighbors.
    - `:sfft` the `:fft` over a ``k + \\log n`` random sample
    - `:sdnet` the `:dnet` over a ``k + \\log n`` random sample
    - `:rand` selects the set of random points along the dataset.
    - array of vectors (centers)

`maxiters` is also a value working when `initial` is a clustering strategy. Please see `KCenters.kcenters` for more details.
"""

function DeloneInvIndex(dist::PreMetric, X::AbstractVector, kcenters_::ClusteringData; ksearch=3, k=10)
    k = length(kcenters_.centers)
    dmax = zeros(Float32, k)
    lists = [UInt32[] for i in 1:k]

    for i in eachindex(kcenters_.codes)
        code = kcenters_.codes[i]
        d = kcenters_.distances[i]
        push!(lists[code], i)
        dmax[code] = max(dmax[code], d)
    end

    C = ExhaustiveSearch(dist, kcenters_.centers)
    opts = DeloneInvIndexOptions(length(kcenters_.codes), ksearch)
    DeloneInvIndex(dist, X, C, lists, dmax, KnnResult(k), opts)
end

function DeloneInvIndex(dist::PreMetric, X::AbstractVector; numcenters=0, ksearch=3, initial=:dnet, maxiters=7, k=10)
    numcenters = numcenters == 0 ? ceil(Int, sqrt(length(X))) : numcenters
    centers = kcenters(dist, X, numcenters; initial=initial, maxiters=maxiters)
    index = DeloneInvIndex(dist, X, centers; ksearch=ksearch, k=k)
end


"""
    search(index::DeloneInvIndex, q::T, res::KnnResult)

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(index::DeloneInvIndex, q, res::KnnResult)
    #cres = KnnResult(index.ksearch)
    cres = search(index.centers, q)

    for i in 1:length(cres)
        (id_, dist_) = cres[i]
        @inbounds for i in index.lists[id_]
            d = SimilaritySearch.evaluate(index.dist, q, index.db[i])
            push!(res, i, d)
        end
    end

    res
end


"""
    optimize!(perf::Performance, index::DeloneInvIndex, recall=0.9; numqueries=128, verbose=false, k=10)

Tries to configure `index` to achieve the specified recall for fetching `k` nearest neighbors.
"""
function optimize!(perf::Performance, index::DeloneInvIndex, recall=0.9; numqueries=128, verbose=false, k=10)
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
end
