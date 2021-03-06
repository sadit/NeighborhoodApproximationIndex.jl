# Similarity Search Indexes based on Neighborhood Approximation

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sadit.github.io/NeighborhoodApproximationIndex.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/NeighborhoodApproximationIndex.jl/dev)
[![Build Status](https://travis-ci.com/sadit/NeighborhoodApproximationIndex.jl.svg?branch=master)](https://travis-ci.com/sadit/NeighborhoodApproximationIndex.jl)
[![Build Status](https://github.com/sadit/NeighborhoodApproximationIndex.jl/workflows/CI/badge.svg)](https://github.com/sadit/NeighborhoodApproximationIndex.jl/actions)
[![Coverage](https://codecov.io/gh/sadit/NeighborhoodApproximationIndex.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sadit/NeighborhoodApproximationIndex.jl)


## Installation

```julia
using Pkg
Pkg.add("NeighborhoodApproximation")

```

## Usage

```julia
using SimilaritySearch
using NeighborhoodApproximationIndex
function create_dataset(n, dim)
    [rand(Float32, dim) for i in 1:n]
end

X = create_dataset(10_000, 8)
queries = create_dataset(100, 8)
napp = fit(DeloneInvIndex, l2_distance, X; numcenters=128, initial=:rand, maxiters=3, region_expansion=3)
```

See examples at [examples directory](https://github.com/sadit/NeighborhoodApproximationIndex.jl/tree/main/examples) of this repository. These are [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebooks, and probably you must install it in you wanna take a look.




## About the methods in this package
This package contains some implementations of approximate similarity search methods based on neighborhood approximation, similar to those found in 

```
Edgar Chavez, Mario Graff, Gonzalo Navarro, Eric S. Tellez:
Near neighbor searching with K nearest references. Inf. Syst. 51: 43-61 (2015)

Eric S. Tellez, Edgar Chavez, Gonzalo Navarro: Succinct nearest neighbor search. Inf. Syst. 38(7): 1019-1030 (2013)

Eric S. Tellez, Edgar Chavez, Gonzalo Navarro: Succinct nearest neighbor search. SISAP 2011: 33-40 (2011)

```

I am rewritting some of this methods in Julia. Therefore, by now, you will not find a complete set of features (like compact datastructures) or a broad exploration in the distance functions in the mapped space. The precise implementations with compression can be found in the old C# library [natix](https://github.com/sadit/natix).
