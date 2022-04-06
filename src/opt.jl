# This file is a part of NeighborhoodApproximationIndex.jl

using SearchModels, Random, StatsBase
import SearchModels: combine, mutate
import SimilaritySearch: optimize!, MinRecall, ParetoRecall, ParetoRadius, ErrorFunction
export optimize!, KnrOptSpace

@with_kw struct KnrOptSpace <: AbstractSolutionSpace
    ksearch = 3:3:15
    ksearch_scale = (s=1.2, p1=0.8, p2=0.8, lower=1, upper=128)
end

Base.hash(c::KnrOpt) = hash(c.ksearch)
Base.isequal(a::KnrOpt, b::KnrOpt) = a.ksearch == b.ksearch
Base.eltype(::KnrOptSpace) = KnrOpt
Base.rand(space::KnrOptSpace) = KnrOpt(rand(space.ksearch))
 
combine(a::KnrOpt, b::KnrOpt) = KnrOpt((a.ksearch + b.ksearch) ÷ 2)
mutate(sp::KnrOptSpace, a::KnrOpt, iter) = KnrOpt(SearchModels.scale(a.ksearch; sp.ksearch_scale...))

function eval_config(index::KnrIndex, gold, knnlist::Vector{KnnResult}, queries; ksearch, verbose)
    n = length(index)
    nt = Threads.nthreads()
    vmin = Vector{Float64}(undef, nt)
    vmax = Vector{Float64}(undef, nt)
    vacc = Vector{Float64}(undef, nt)
    covradius = Vector{Float64}(undef, length(knnlist))
    pools = getpools(index)
    
    function lossfun(conf)
        vmin .= typemax(eltype(vmin))
        vmax .= typemin(eltype(vmax))
        vacc .= 0.0
        
        searchtime = @elapsed begin
            Threads.@threads for i in 1:length(queries)
                _, v = search(index, queries[i], reuse!(knnlist[i], ksearch); pools, ksearch=conf.ksearch)
                ti = Threads.threadid()
                vmin[ti] = min(v, vmin[ti])
                vmax[ti] = max(v, vmax[ti])
                vacc[ti] += v
            end
        end

        v = minimum(vmin), sum(vacc)/length(knnlist), maximum(vmax)
        # we assume 0 (res's shift=>0), this is true for the BS algorithm, and typical algorithms
        # popshift! is used with other purposes and must to be applied to the result set
        
        for i in eachindex(knnlist)
            covradius[i] = maximum(knnlist[i])
        end
        rmin, rmax = extrema(covradius)
        ravg = mean(covradius)

        recall = if gold !== nothing
            macrorecall(gold, [Set(res.id) for res in knnlist])
        else
            nothing
        end

        verbose && println(stderr, "eval_config> config: $conf, searchtime: $searchtime, recall: $recall, length: $(length(index))")
        (visited=v, radius=(rmin, ravg, rmax), recall=recall, searchtime=searchtime/length(knnlist))
    end
end

_kfun(x) = 1.0 - 1.0 / (1.0 + x)

"""
    optimize!(
        index::KnrIndex,
        kind::ErrorFunction=ParetoRecall();
        queries=nothing,
        queries_ksearch=10,
        queries_size=64,
        initialpopulation=8,
        verbose=false,
        space=KnrOptSpace(),
        params=SearchParams(; maxpopulation=8, bsize=4, mutbsize=8, crossbsize=2, tol=-1.0, maxiters=8, verbose),
    )

Tries to configure the `index` to achieve the specified performance (`kind`). The optimization procedure is an stochastic search over the configuration space yielded by `kind` and `queries`.

# Arguments
- `index`: the `KnrIndex` to be optimized
- `kind`: The kind of optimization to apply, it can be `ParetoRecall()`, `ParetoRadius()` or `MinRecall(r)` where `r` is the expected recall (0-1, 1 being the best quality but at cost of the search time)

# Keyword arguments

- `queries`: the set of queries to be used to measure performances, a validation set. It can be an `AbstractDatabase` or nothing.
- `queries_ksearch`: the number of neighbors to retrieve for `queries`
- `queries_size`: if `queries===nothing` then a sample of the already indexed database is used, `queries_size` is the size of the sample.
- `initialpopulation`: the initial sample for the optimization procedure
- `space`: defines the search space
- `params`: the parameters of the solver, see [`search_models` function from `SearchModels.jl`](https://github.com/sadit/SearchModels.jl) package for more information.
- `verbose`: controls if the procedure is verbose or not
"""
function optimize!(
            index::KnrIndex,
            kind::ErrorFunction=ParetoRecall();
            queries=nothing,
            queries_ksearch=10,
            queries_size=64,
            initialpopulation=8,
            verbose=false,
            space=KnrOptSpace(),
            params=SearchParams(; maxpopulation=8, bsize=4, mutbsize=8, crossbsize=2, tol=-1.0, maxiters=8, verbose),
    )

    if queries === nothing
        sample = rand(1:length(index), queries_size) |> unique
        queries = SubDatabase(index.db, sample)
    end

    knnlist = [KnnResult(queries_ksearch) for i in eachindex(queries)]
    gold = nothing
    if kind isa ParetoRecall || kind isa MinRecall
        db = @view index.db[1:length(index)]
        seq = ExhaustiveSearch(index.dist, db)
        searchbatch(seq, queries, knnlist; parallel=true)
        gold = [Set(res.id) for res in knnlist]
    end

    M = Ref(0.0)
    R = Ref(0.0)

    function inspect_pop(space, params, population)
        if M[] == 0.0
            for (c, p) in population
                M[] = max(p.visited[end], M[])
                R[] = max(p.radius[end], R[])
            end
        end
    end

    error_function = eval_config(index, gold, knnlist, queries; ksearch=queries_ksearch, verbose)

    function geterr(p)
        cost = p.visited[2] / M[]
        if kind isa ParetoRecall 
            cost^2 + (1.0 - p.recall)^2
        elseif kind isa MinRecall
            p.recall < kind.minrecall ? 3.0 - 2 * p.recall : cost
        else
            _kfun(cost) + _kfun(p.radius[2] / R[])
        end
    end
    
    bestlist = search_models(
        error_function,
        space, 
        initialpopulation,
        params;
        inspect_population=inspect_pop,
        geterr=geterr)
    
    config, perf = bestlist[1]
    verbose && println(stderr, "== finished opt. $(typeof(index)): search-params: $(params), opt-config: $config, perf: $perf, length=$(length(index))")
    index.opt.ksearch = config.ksearch
    bestlist
end
