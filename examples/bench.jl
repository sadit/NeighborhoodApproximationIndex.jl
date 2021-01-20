using SimilaritySearch, NeighborhoodApproximationIndex

function create_dataset(n, dim)
    [rand(Float32, dim) for i in 1:n]
end

function run(index, queries, k)
    @info ("========>", string(index), length(queries), k)
    @time for q in queries
        search(index, q, k)
    end
end


function main()
	dim = 8
	X = create_dataset(100_000, dim)
	queries = create_dataset(100, dim)
	dist = SqL2Distance()
    napp = DeloneInvIndex(dist, X; initial=:rand, maxiters=3, ksearch=1)
    seq = ExhaustiveSearch(dist, X)
    @show size(X), size(queries), dist
    k = 10
    run(seq, queries, k)
    run(napp, queries, k)
end

main()