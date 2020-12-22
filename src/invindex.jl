# This file is a part of KCenters.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import StatsBase: fit, predict
using StatsBase
using SimilaritySearch
import SimilaritySearch: search, optimize!
using KCenters

using JSON
export DeloneInvIndex, fit, predict

mutable struct DeloneInvIndex{T} <: Index
    db::Vector{T}
    centers::Index
    lists::Vector{Vector{UInt32}}
    dmax::Vector{Float32}
    n::Int
    region_expansion::Int
end

"""
    fit(::Type{DeloneInvIndex}, dist::Function, X::AbstractVector{T}; numcenters=128, region_expansion=3, initial=:dnet, maxiters=7) where T

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
function fit(::Type{DeloneInvIndex}, dist::Function, X::AbstractVector{T}; numcenters=128, region_expansion=3, initial=:dnet, maxiters=7) where T
    centers = kcenters(dist, X, numcenters; initial=initial, maxiters=maxiters)
    index = fit(DeloneInvIndex, X, centers; region_expansion=region_expansion)
end

"""
    fit(::Type{DeloneInvIndex}, X::AbstractVector{T}, kcenters_::NamedTuple; region_expansion=3) where T

Creates a DeloneInvIndex with a given clustering data (`kcenters_`).

"""
function fit(::Type{DeloneInvIndex}, X::AbstractVector{T}, kcenters_::NamedTuple; region_expansion=3) where T
    k = length(kcenters_.centroids)
    dmax = zeros(Float32, k)
    lists = [UInt32[] for i in 1:k]

    for i in eachindex(kcenters_.codes)
        code = kcenters_.codes[i]
        d = kcenters_.distances[i]
        push!(lists[code], i)
        dmax[code] = max(dmax[code], d)
    end

    C = fit(Sequential, kcenters_.centroids)
    DeloneInvIndex(X, C, lists, dmax, length(kcenters_.codes), region_expansion)
end

"""
    search(index::DeloneInvIndex{T}, dist::Function, q::T, res::KnnResult) where T

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(index::DeloneInvIndex{T}, dist::Function, q::T, res::KnnResult) where T
    cres = search(index.centers, dist, q, KnnResult(index.region_expansion))
    for c in cres
        @inbounds for i in index.lists[c.id]
            d = dist(q, index.db[i])
            push!(res, i, d)
        end
    end

    res
end


"""
    optimize!(index::DeloneInvIndex{T}, dist::Function, recall=0.9; k=10, num_queries=128, perf=nothing, verbose=false) where T

Tries to configure `index` to achieve the specified recall for fetching `k` nearest neighbors. Notice that if `perf` is not given then
the index will use dataset's items and therefore it will adjust for them.
"""
function optimize!(index::DeloneInvIndex{T}, dist::Function, recall=0.9; k=10, num_queries=128, perf=nothing, verbose=false) where T
    verbose && println("KCenters.DeloneInvIndex> optimizing for recall=$(recall)")
    if perf === nothing
        perf = Performance(index.db, dist; expected_k=k, num_queries=num_queries)
    end

    index.region_expansion = 1
    p = probe(perf, index, dist)

    while p.recall < recall && index.region_expansion < length(index.lists)
        index.region_expansion += 1
        verbose && println("KCenters.DeloneInvIndex> optimize! step region_expansion=$(index.region_expansion), performance $(JSON.json(p))")
        p = probe(perf, index, dist)
    end

    verbose && println("KCenters.DeloneInvIndex> reached performance $(JSON.json(p))")
    index
end
